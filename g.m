function g_z = g(z, activation)
    if activation == "sigmoid"
        g_z = 1./(1+exp(-z));
    elseif activation == "relu"
        g_z = max(0, z);
    end
end