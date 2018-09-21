function [updated_w_1, updated_b1, updated_w_2, updated_b2] = update_wb(x, y, w_1, w_2, b1, b2, eta, activation, task)
    [m, d] = size(x);
    [~, d1] = size(w_1);
    coefficient = 0;
    
    if task == "regression"
        coefficient = 2;
        common_term = transpose(fwb(x, w_1, w_2, b1, b2, activation)) - y;
    elseif task == "classification"
        coefficient = 1;
        common_term = transpose(g(fwb(x, w_1, w_2, b1, b2, activation), activation)) - y;
    end
    
    db2 = coefficient * mean(transpose(common_term), 2);
    
    dw_2 = coefficient * mean(transpose(repmat(common_term, 1, d1) .* transpose(a1(x, w_1, b1, activation))), 2);
    
    new_term = repmat(common_term, 1, d1) .* transpose(repmat(w_2, 1, m) .* dg(z1(x, w_1, b1), activation));
    db1 = coefficient * mean(new_term, 1);
    
    dw_1 = zeros(d1, d);
    for i = 1:m
        dw_1 = dw_1 + transpose(new_term(i, :))*x(i, :)/m;
    end
    dw_1 = coefficient * dw_1';
    
    updated_w_1 = w_1 - eta*dw_1;
    updated_b1 = b1 - eta*db1;
    updated_w_2 = w_2 - eta*dw_2;
    updated_b2 = b2 - eta*db2;
end