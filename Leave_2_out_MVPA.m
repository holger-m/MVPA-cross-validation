function [Accu_mean, Accu_mat, train_bin_mat] = Leave_2_out_MVPA(Y, X, c_vec)


Case_ind = find(Y == 1);
Cont_ind = find(Y == -1);


train_bin_mat = false(size(Y,1), size(Case_ind,1)*size(Cont_ind,1));

column_count = 0;

for case_subj = 1:size(Case_ind,1)
    
    for cont_subj = 1:size(Cont_ind,1)
        
        column_count = column_count+1;
        
        case_subj_ind = Case_ind(case_subj,1);
        cont_subj_ind = Cont_ind(cont_subj,1);
        
        train_ind = true(size(Y,1),1);
        train_ind([case_subj_ind; cont_subj_ind],1) = false;
        
        train_bin_mat(:,column_count) = train_ind;
        
    end
    
end


Accu_mat = false(size(Case_ind,1)*size(Cont_ind,1), size(c_vec,2), 2);

for model_no = 1:size(Case_ind,1)*size(Cont_ind,1)

    train_ind = train_bin_mat(:,model_no);

    X_train = X(train_ind,:);
    X_test = X(~train_ind,:);
    Y_train = Y(train_ind,:);
    Y_test = Y(~train_ind,:);

    for c_ind = 1:size(c_vec,2)

        c_value = c_vec(1,c_ind);

        clear model
        
        model = svmtrain(Y_train, X_train, ['-t 0 -c ',num2str(c_value),' -q']); %#ok<SVMTRAIN>

        Y_pred = svmpredict(Y_test, X_test, model, '-q');
            
        Y_accu = Y_pred==Y_test;

        Accu_mat(model_no,c_ind,:) = Y_accu;

    end 
    
end

Accu_mean = mean(mean(Accu_mat,3),1)';
