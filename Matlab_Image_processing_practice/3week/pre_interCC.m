function [img_out] = pre_interCC(img, N)

img = double(img);
img_out = interpCC(img);
img_out = permute(img_out,[2,1,3]);
img_out = interpCC(img_out);
img_out = permute(img_out,[2,1,3]);



    function [img_output] = interpCC(img_input)
    [H, W] = size(img_input);
    img_output = zeros(H, W*N);
    img_output(:,1:N:end) = img_input;

    for i  = 1: H
        for j = 1:N:W*N

            x2 = img_output(i,j);

            if j - N < 0 
                x1 = x2;
            else
                x1 = img_output(i,j-N);
            end

            if j + N > W*N 
                x3 = x2;
            else
                x3 = img_output(i,j+N);
            end


            if j + 2*N > W*N 
                x4 = x3;
            else
                x4 = img_output(i,j+2*N);
            end

            for k = 1 : N-1
                img_output(i, j+k) = beta_func(k+N,N)*x1 ...
                                            +beta_func(k,N)*x2 ... 
                                            +beta_func(k-N,N)*x3 ... 
                                            +beta_func(k-2*N,N)*x4;
            end
        end
    end

        
    end

    function [out] = beta_func(x, n)

    a = -0.5;
    x = x/n;

    if abs(x) <1
        out = (a+2)*abs(x)^3 - (a+3)*abs(x)^2 + 1;
    elseif abs(x) < 2
        out = a*abs(x)^3 - 5*a*abs(x)^2 + 8*a*abs(x)-4*a;
    else
        out = 0;
    end

    end

end