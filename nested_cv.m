function [overall_accu] = nested_cv(L2SO_Accu_mat, L2SO_train_bin_mat, L4SO_Accu_mat, L4SO_train_bin_mat)

[L2SO_test_ind_a, ~] = find(~L2SO_train_bin_mat);

L2SO_test_ind = reshape(L2SO_test_ind_a, 2, size(L2SO_train_bin_mat,2));

[L4SO_test_ind_a, ~] = find(~L4SO_train_bin_mat);

L4SO_test_ind = reshape(L4SO_test_ind_a, 4, size(L4SO_train_bin_mat,2));


Nested_accu_vec = NaN(size(L2SO_Accu_mat,1),1);

for L2SO_model_no = 1:size(L2SO_Accu_mat,1)
    
    L2SO_test_ind_curr = L2SO_test_ind(:,L2SO_model_no);
    
    L4SO_nested_ind = any(L4SO_test_ind == L2SO_test_ind_curr(1,1)) & any(L4SO_test_ind == L2SO_test_ind_curr(2,1));
    
    L4SO_Accu_mat_nested = L4SO_Accu_mat(L4SO_nested_ind,:,:);
    
    L4SO_test_ind_nested = L4SO_test_ind(:,L4SO_nested_ind)';
    
    L4SO_Accu_mat_nested_clean = false(size(L4SO_Accu_mat_nested,1), size(L4SO_Accu_mat_nested,2), 2);
    
    for L4SO_model_no = 1:size(L4SO_Accu_mat_nested,1)
        
        L4SO_test_ind_nested_curr = L4SO_test_ind_nested(L4SO_model_no,:);
        
        include_subj_bin = L4SO_test_ind_nested_curr ~= L2SO_test_ind_curr(1,1) & L4SO_test_ind_nested_curr ~= L2SO_test_ind_curr(2,1);
        
        L4SO_Accu_mat_nested_clean(L4SO_model_no,:,:) = L4SO_Accu_mat_nested(L4SO_model_no,:,include_subj_bin);
        
    end
    
    L4SO_Accu_mat_nested_clean_mean = mean(mean(L4SO_Accu_mat_nested_clean,3),1);
    
    [~, c_best_ind] = max(L4SO_Accu_mat_nested_clean_mean);
    
    Nested_accu_vec(L2SO_model_no,1) = mean(L2SO_Accu_mat(L2SO_model_no, c_best_ind,:),3);    
    
end

overall_accu = mean(Nested_accu_vec);

