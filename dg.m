function dg_z = dg(z, activation)
    if activation == "sigmoid"
        %dg_z = g(z, activation) .* (1 - g(z, activation));
        dg_z = exp(-z)./(1+exp(-z)).^2;
    elseif activation == "relu"
        dg_z = z > 0;
    end
end