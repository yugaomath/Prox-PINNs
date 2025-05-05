from Solvers.Module import Solver
###############################################################################
class Example():

    def __init__(self, idx:str, path:str, solver_name:str):
        
        self.idx = idx
        self.path = path
        self.solver_name = solver_name

    def run(self, solver:Solver, load_type:str, status:str):
        
        if status == 'train':
            
            print('\n*****************************************************************************')
            print('******************************* Start Training ******************************')
            print('*****************************************************************************')
            
            solver.train(save_path = self.path+f'{self.solver_name}_{self.idx}/', 
                         load_type = load_type)
            
        elif status == 'pred':
            
            print('\n*****************************************************************************')
            print('******************************* Start Prediction ****************************')
            print('*****************************************************************************')
            
            solver.predict(load_path = self.path+f'{self.solver_name}_{self.idx}/', 
                           load_type = load_type)
            
        elif status =='plot':
            
            print('\n*****************************************************************************')
            print('******************************* Start Plot Figure ***************************')
            print('*****************************************************************************')
            
            solver.plot_fig(load_path = self.path+f'{self.solver_name}_{self.idx}/')
            
        else:
            
            print('Input Status is worng!!!')