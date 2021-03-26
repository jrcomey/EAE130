alpha = linspace(0, 10, 10)';
beta = linspace(0, 10, 10)';
c = linspace(0, 1, 10)';

abc = [alpha, beta, c]

[amesh, bmesh] = meshgrid(alpha, beta)

for i = 1:10
    for k = 1:10
        cmesh(i, k) = 0;
        
    end
end