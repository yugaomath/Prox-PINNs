###############################################################################
import os
import numpy as np 
import torch
import time
import scipy.io
import matplotlib.pyplot as plt
###############################################################################
from Network.Network import Model
from Problems.Module import Problem
from Utils.Error import Error
import Solvers.Module as Module
###############################################################################
class Solver(Module.Solver):

    def __init__(self, problem: Problem, **kwargs):
        '''
        Input:
            problem: 
            N_bd_each_face: the number of points on the boundary (each face)
            N_int: the number of meshgrids (or meshsize) for computing integration
            maxIter: the maximum of iterations
            lr: the learning rate
            model_type: 'FeedForward'
            data_type: {'numpy', 'torch'}
            kwargs: 
        '''
        #
        self.problem = problem
        #
        self.N_int = kwargs['N_int']
        self.N_bd_each_face = kwargs['N_bd_each_face']
        
        self.lr_adam = kwargs['lr_adam']
        self.lr_lbfgs = kwargs['lr_lbfgs']
        self.Iter_adam = kwargs['Iter_adam']
        self.Iter_lbfgs = kwargs['Iter_lbfgs']
        
        self.model_type = kwargs['model_type']
        self.numpy_dtype = kwargs['numpy_dtype']
        self.torch_dtype = kwargs['torch_dtype']
        
        # Other parameter setting
        self.device = kwargs['device']
        self.nn_in = kwargs['NN_in']
        self.nn_out = kwargs['NN_out']
        self.lr_Decay = kwargs['lr_Decay']
        self.loss_weight = kwargs['loss_weight']
        self.topK = kwargs['topK']
        self.int_method = kwargs['int_method']
        self.hidden_n = kwargs['hidden_width']
        self.hidden_l = kwargs['hidden_layer']
        self.activation = kwargs['activation']

    def _load(self, model:torch.nn.Module, load_path:str, load_type:str)->None:
        '''
        Input:
            model: torch.nn.Module
            load_path: the path of the trained model to be loaded
            load_type: 'model_best_error', 'model_best_loss', 'model_final'
        '''
        
        model_dict = torch.load(load_path+f'{load_type}.pth', map_location=torch.device(self.device))
        model.load_state_dict(model_dict['model'])
        

    def _save(self, model:torch.nn.Module, save_path:str, save_type:str)->None:
        '''
        Input: 
            model: torch.nn.Module
            save_path: the path for the trained model to be saved
            save_type: 'model_final', 'model_best_error', 'model_best_loss'
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save the loss and error if save_type=='model_final'
        if save_type=='model_final':
            dict_loss = {}
            dict_loss['loss'] = self.loss_list
            dict_loss['l2_error'] = self.l2_error_list
            dict_loss['linf_error'] = self.linf_error_list
            dict_loss['time'] = self.time_list
            scipy.io.savemat(save_path+'loss_error_saved.mat', dict_loss)
        # save the trained model
        model_dict = {'model':model.state_dict()}
        torch.save(model_dict, save_path+f'{save_type}.pth')


    def get_net(self)->None:
        '''
        Get the network model
        '''
        ##### The model structure
        kwargs = {'d_in': self.nn_in,
                  'd_out': self.nn_out,
                  'h_size': self.hidden_n,
                  'l_size': self.hidden_l,
                  'activation': self.activation,
                  'lb': self.problem.lb.to(self.device), 
                  'ub': self.problem.ub.to(self.device),
                  'pde_domain': self.problem._domain,
                  'R': torch.tensor(self.problem._R, dtype=self.torch_dtype).to(self.device),
                  'fun_u': self.problem.fun_u}
        
        self.model = Model(self.model_type, self.device, dtype=self.torch_dtype).get_model(**kwargs)
        ###### The optimizer
        self.adam_optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.lr_adam}])
        self.lbfgs_optimizer = torch.optim.LBFGS([{'params': self.model.parameters(), 'lr': self.lr_lbfgs}])
        ####### The scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.adam_optimizer, 1, gamma=(1. - self.lr_Decay/self.Iter_adam), last_epoch=-1)


    def get_loss(self, data_point:dict, **args):
        '''
        Input:
            data_point: dict={'x_in', 'x_bd_list'}
        Output:
            loss_train:
            loss_all: [loss_in, loss_bd]
        '''
        loss_all = []
        #######################################################################
        ########### Residual inside the domain
        #######################################################################

        eq = self.problem.evi_pinn(model=self.model, x=data_point['x_in'].to(self.device))
        eq = eq**2
        
        try:
            eq_topk, index = torch.topk(eq, k=self.topK, dim=0)
        except:
            eq_topk = eq
        
        #######################################################################
        
        loss_in = torch.mean(eq_topk)
        x_bd_list = [item.to(self.device) for item in data_point['x_bd_list']]
        cond_bd = self.problem.fun_bd_err(model=self.model, x_list=x_bd_list)
        loss_bd = torch.mean( cond_bd**2 ) 
        loss_all.append([loss_in.detach(), loss_bd.detach()])
        
        if self.model_type == 'FeedForward_Constraint':
            loss_train = loss_in
            
        else:
            loss_train = (loss_in * self.loss_weight['eq'] + loss_bd * self.loss_weight['bd'])
        #######################################################################
        
        return loss_train, loss_all
    
    ###########################################################################
    
    def _print(self, iter:int, loss_train:float, loss_eq:float, loss_bd:float, re_l2_err:float, re_linf_err:float)->None:
        print(f" iter: {iter}, loss_train:{loss_train:.3e}, loss_eq: {loss_eq:.3e}, loss_bd: {loss_bd:.3e}; re_l2_err: {re_l2_err:.3e}, re_linf_err: {re_linf_err:.3e}")
        
    ###########################################################################


    def train(self, save_path:str, load_path:str=None, load_type:str=None)->None:
        '''
        Input: 
            save_path: 
            load_path: path for loading trained model
            load_type: 'model_best_l2_error', 'model_best_linf_error','model_best_loss', 'model_final'
        '''
        t_start = time.time()
        
        #######################################################################
        
        self.get_net()
        try:
            self._load(self.model, save_path, load_type)
            print('  A trained model has been loaded ......\n')
        except:
            print('  Started with a new model ......\n ')
            
        #######################################################################
        
        u_test, x_test = self.problem.get_test()
        #######################################################################

        data_train = self.problem.get_point_disk(N_xin = self.N_int, 
                                                 N_xbd_each_face = self.N_bd_each_face)
        
        #######################################################################
        ########################### Start training process ####################
        #######################################################################
        
        iter = 0
        best_l2_err, best_linf_err, best_loss = 1e10, 1e10, 1e10
        self.time_list, self.loss_list, self.l2_error_list, self.linf_error_list = [], [], [], []
        
        #######################################################################

        print(' Optimized by Adam:')
        for iter in range(self.Iter_adam):
            
            ###################################################################
            
            loss_train, loss_all = self.get_loss(data_train)
            
            ###################################################################
            
            self.loss_list.append(loss_train.item())
            self.time_list.append(time.time()-t_start)
            
            ###################################################################
            
            l2_err, linf_err, _ = Error().L2_and_Linf_error(model=self.model, x=x_test.to(self.device), u=u_test.to(self.device))
            self.l2_error_list.append(l2_err.item())
            self.linf_error_list.append(linf_err.item())
            
            ###################################################################
            
            if l2_err.item() < best_l2_err:
                best_l2_err = l2_err.item()
                self._save(self.model, save_path, save_type='model_best_l2_error')
                
            if linf_err.item() < best_linf_err:
                best_linf_err = linf_err.item()
                self._save(self.model, save_path, save_type='model_best_linf_error')
                
            if loss_train.item() < best_loss:
                best_loss = loss_train.item()
                self._save(self.model, save_path, save_type='model_best_loss')
            # 
            if (iter+1)%100 == 0:
                loss = loss_all[0]
                self._print(iter+1, loss_train.item(), loss[0].item(), loss[1].item(), l2_err.item(), linf_err.item())
                
            ###################################################################
            
            self.adam_optimizer.zero_grad()
            loss_train.backward()
            self.adam_optimizer.step()
            self.scheduler.step()
            iter += 1
            
            ###################################################################
        
        #######################################################################
        print('\n Optimized by L-BFGS:')
        for epoch in range(self.Iter_lbfgs):
            
            def closure():
                self.lbfgs_optimizer.zero_grad()
                loss_train, loss_all = self.get_loss(data_train)
                loss_train.backward()
                return loss_train
            
            loss_train = closure()
            
            self.loss_list.append(loss_train.item())
            self.time_list.append(time.time()-t_start)
            
            l2_err, linf_err, _ = Error().L2_and_Linf_error(model=self.model, x=x_test.to(self.device), u=u_test.to(self.device))
            self.l2_error_list.append(l2_err.item())
            self.linf_error_list.append(linf_err.item())
               
            ###################################################################
               
            if l2_err.item() < best_l2_err:
                best_l2_err = l2_err.item()
                self._save(self.model, save_path, save_type='model_best_l2_error')
                   
            if linf_err.item() < best_linf_err:
                best_linf_err = linf_err.item()
                self._save(self.model, save_path, save_type='model_best_linf_error')
                   
            if loss_train.item() < best_loss:
                best_loss = loss_train.item()
                self._save(self.model, save_path, save_type='model_best_loss')
                
            self.lbfgs_optimizer.step(closure)
        
            if (epoch + 1) % 10 == 0:
                loss_train = closure()
                print(f" iter: {epoch+1}, loss_train:{loss_train.item():.3e}; re_l2_err: {l2_err.item():.3e}, re_linf_err: {linf_err.item():.3e}")
        
        #######################################################################
        
        print(f'\n  The total training time is: {time.time()-t_start:.4f}s')
        
        #######################################################################
        
        self._save(self.model, save_path, save_type='model_final')
        print(f'\n  Trainging results have been saved in:\n  {load_path}')
        

        #######################################################################
        
    def predict(self, load_path:str, load_type:str)->None:
        
        #######################################################################
        '''
        Input:
            load_path: the path of the trained model
            load_type: 'model_best_l2_error', 'model_best_linf_error','model_best_loss', 'model_final'
        '''
        #######################################################################
        print(f'\n  Load the trained model from:\n  {load_path}')
        # load the trained model
        self.get_net()
        self._load(self.model, load_path, load_type)
        # prediction
        
        x_mesh = np.linspace(-self.problem._R, self.problem._R, 100)
        x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)
        xnp = x_mesh.reshape(-1,1)
        ynp = y_mesh.reshape(-1,1)
        xynp = np.concatenate([xnp, ynp], axis=1)
        x_test = torch.from_numpy(xynp.astype(self.problem._numpy_dtype))
        
        u_test = self.problem.fun_u(x_test)
        with torch.no_grad():
            up_pred = self.model(x_test.to(self.device))
            u_pred = up_pred[:,0:1]
            
        # save result
        dict_test = {}
        dict_test['x_test'] = x_test.detach().cpu().numpy()
        dict_test['u_test'] = u_test.detach().cpu().numpy()
        dict_test['u_pred'] = u_pred.detach().cpu().numpy()
        scipy.io.savemat(load_path+'test_saved.mat', dict_test)
        #
        print(f'\n  Prediction results have been saved in:\n  {load_path}')
        
        #######################################################################
        
    def plot_fig(self, load_path:str)->None:
        
        #######################################################################
        
        print(f'\n  Load the prediction results from:\n  {load_path}')
        
        dict_loss_error = scipy.io.loadmat(load_path+'loss_error_saved.mat')
        
        loss_list = dict_loss_error['loss'][0] 
        l2_err_list = dict_loss_error['l2_error'][0]  
        linf_err_list = dict_loss_error['linf_error'][0] 
        epochs = np.arange(1, len(loss_list)+1)
        
        fig_path = load_path+'/figures/'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        
        plt.figure(1)
        plt.plot(epochs, np.log10(loss_list), 'r--', label = r"$\log_{10}(Loss)$")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(fig_path+'loss.png', dpi = 750, bbox_inches = 'tight')
        plt.show()
        
        plt.figure(2)
        plt.plot(epochs, np.log10(l2_err_list), 'b--', label = r"$\log_{10}(\|u_{\rm exact}-u_{\rm DNN}\|_{L_{2}}/\|u_{\rm exact}\|_{L_{2}})$")
        plt.plot(epochs, np.log10(linf_err_list), 'r--', label = r"$\log_{10}(\|(u_{\rm exact}-u_{\rm DNN})\|_{L_{\infty}}/\|(u_{\rm exact}\|_{L_{\infty}})$")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(fig_path+'error.png', dpi = 750, bbox_inches = 'tight')
        plt.show()
        
        #######################################################################
        
        dict_test = scipy.io.loadmat(load_path+'test_saved.mat')
        
        x_test = dict_test['x_test'] 
        u_test = dict_test['u_test'] 
        u_pred = dict_test['u_pred']
        
        length = int(np.sqrt(x_test.shape[0]))
        x_mesh = x_test[:,0:1].reshape(length, length)
        y_mesh = x_test[:,1:2].reshape(length, length)
        
        u_test = u_test.reshape(length, length)
        u_pred = u_pred.reshape(length, length)
        
        R = self.problem._R
        ind = np.sqrt(x_mesh**2+y_mesh**2)> R
        u_test[ind] = np.nan
        u_pred[ind] = np.nan
        
        fig = plt.figure(3)
        ax1 = fig.add_subplot(111)
        contour1 = ax1.contourf(x_mesh, y_mesh, u_test, levels=100, cmap='viridis')  
        plt.colorbar(contour1)  
        ax1.set_title(r'$u_{\rm exact}$')
        plt.tight_layout()
        plt.savefig(fig_path+'exact.png', dpi = 750, bbox_inches = 'tight')
        plt.show()
        
        fig = plt.figure(4)
        ax2 = fig.add_subplot(111)
        contour2 = ax2.contourf(x_mesh, y_mesh, u_pred, levels=100, cmap='viridis')  
        plt.colorbar(contour2)  
        ax2.set_title(r'$u_{\rm DNN}$')
        plt.tight_layout()
        plt.savefig(fig_path+'pred.png', dpi = 750, bbox_inches = 'tight')
        plt.show()
        
        fig = plt.figure(5)
        ax3 = fig.add_subplot(111)
        contour3 = ax3.contourf(x_mesh, y_mesh, np.abs(u_test-u_pred), levels=100, cmap='viridis')  
        plt.colorbar(contour3)  
        ax3.set_title(r'$|u_{\rm exact}-u_{\rm DNN}|$')
        plt.tight_layout()
        plt.savefig(fig_path+'exact_pred_error.png', dpi = 750, bbox_inches = 'tight')
        plt.show()


        



