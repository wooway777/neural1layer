function fwbx = fwb(x, w_1, w_2, b1, b2, activation)
    fwbx = transpose(w_2) * g(z1(x, w_1, b1), activation) + b2;
end