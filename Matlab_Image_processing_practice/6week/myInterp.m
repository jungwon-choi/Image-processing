function [output] = myInterp(img, N)

img = double(img);

output = interp_sub(img);
output = permute(output, [2 1]);
output = interp_sub(output);
output = permute(output, [2 1]);


    function [out] = interp_sub(input)
        [H, W] =  size(input);
        out = zeros(H, W*N);
        out(:, 1:N:end) = input;
        
        for i = 1 : H
            for j = 1:N:W*N
                
                x1 = out(i,j);
                if j + N > W*N
                    x2 = x1;
                else
                    x2 = out(i,j+N);
                end
                
                for k = 1:N-1
                    out(i,j+k) = beta1(k,N)*x1 ...
                                    +beta1(k-N,N)*x2;
                end
            end
        end
        
    end

    function [out] = beta1(x,n)
        
        x = x/n;
        
        if  0 <= x &&  x <= 1
            out = 1 - x;
        elseif x < 0 && -1 <= x 
            out = 1 + x;
        else
            out = 0;
        end
    end

end
