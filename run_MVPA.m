function [overall_accu_vec, networks_of_interest] = run_MVPA()

networks_of_interest = [[1,4];...  % Selection based on Mohr et al., NI 2018
                        [1,6];...
                        [1,7];...
                        [1,10];...
                        [4,6];...
                        [6,6];...
                        [6,7];...
                        [6,9]];

c_vec = 2.^(-5:1);

overall_accu_vec = NaN(size(networks_of_interest,1),1);

for model_count = 1:size(networks_of_interest,1)
    
    mod_1 = networks_of_interest(model_count,1); 
    mod_2 = networks_of_interest(model_count,2); 
    
    clear X
    clear Y
    
    load(strcat('.\Connectivity_data\Conn_data_',num2str(mod_1,'%02.0f'),'_',num2str(mod_2,'%02.0f'),'.mat')); %#ok<LOAD>
    
    [L2SO_Accu_mean, L2SO_Accu_mat, L2SO_train_bin_mat] = Leave_2_out_MVPA(Y, X, c_vec); %#ok<ASGLU>
    [L4SO_Accu_mean, L4SO_Accu_mat, L4SO_train_bin_mat] = Leave_4_out_MVPA(Y, X, c_vec); %#ok<ASGLU>
    
    overall_accu = nested_cv(L2SO_Accu_mat, L2SO_train_bin_mat, L4SO_Accu_mat, L4SO_train_bin_mat);
    
    overall_accu_vec(model_count,1) = overall_accu;
    
end

