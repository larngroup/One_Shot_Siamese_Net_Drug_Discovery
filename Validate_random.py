# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:33:21 2020

@author: luist
"""

class Validate_random:
    
    
    def validate_random(n_iter,loader, time, siamese_net,batch_size,t_start,n_val,best_val, evaluate_every, loss_every, n):
        
        best_train = -1        
        best_val = -1
        
        random_accs, train_accs = [], []
        
        for i in range(1, n_iter+1):
            
            (inputs,targets) = loader.batch_function(batch_size)
            loss = siamese_net.train_on_batch(inputs, targets)
            
            if i % evaluate_every == 0:
                print("\n ------------- \n")
                print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
                print("Train Loss: {0}".format(loss)) 
                val_acc = loader.random_test(n, n_val)
                
                train_acc = loader.oneshot_test(siamese_net, n, n_val, s = 'train')
  
    #            siamese_net.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
                
    
                if val_acc >= best_val:
                    print("Current best val: {0}, previous best: {1}".format(val_acc, best_val))
                    best_val = val_acc
                    
                if train_acc >= best_train:
                    print("Current best train: {0}, previous best: {1}".format(train_acc, best_train))
                    best_train = train_acc
    
                    
            if i % loss_every == 0:
                print("iteration {}, training loss: {:.2f},".format(i,loss))
                print("Current best val: {0}".format(best_val)," - N:", n)
                print("Current best train: {0}".format(best_train)," - N:", n)

                print("Tempo decorrido:", (time.time()-t_start)/60.0)
                
                
        print("The final best accuracy value (validation): {0}".format(best_val)," - N:", n)    
        print("The final best accuracy value (training): {0}".format(best_train)," - N:", n) 
   
        random_accs.append(best_val)
        train_accs.append(best_train)
        
        return random_accs, train_accs