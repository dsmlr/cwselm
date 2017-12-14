function d=kdist2(X,model)

    %==========================================================================
    % KDIST2 Computes squared distance between vectors in kernel space.
    %
    % Synopsis:
    %  d = kdist2(X,model)
    %
    % Description:
    %  It computes distance between vectors mapped into the feature 
    %  space induced by the kernel function (model.options.ker,
    %  model.options.arg). The distance is computed between images
    %  of vectors X [dim x num_data] mapped into feature space
    %  and a point in the feature space given by model:
    %
    %   d(i) = kernel(X(:,i),X(:,i)) 
    %          - 2*kernel(X(:,i),models.sv.X)*model.Alpha + b,
    %
    %  where b [1x1] is assumed to be equal to 
    %   model.b = model.Alpha'*kernel(model.sv.X)*model.Alpha.
    %
    % Input:
    %  X [dim x num_data] Input vectors.
    %  model [struct] Deternines a point of the feature space:
    %   .Alpha [nsv x 1] Multipliers.
    %   .sv.X [dim x nsv] Vectors.
    %   .b [1x1] Bias.
    %   .options.ker [string] Kernel identifier (see 'help kernel').
    %   .options.arg [1 x nargs] Kernel argument(s).
    %
    % Output:
    %  d [num_data x 1] Squared distance between vectors in the feature space.
    %
    %==========================================================================
    % January 13, 2009
    % Implemented by Daewon Lee
    % WWW: http://sites.google.com/site/daewonlee/
    %==========================================================================

    [~,num_data] = size(X);
    
    if strcmp(model.options.ker,'sim')
        [ n , a , b , c , d ] = FindComponent( X' , X' );
        x2 = diag(Measures( n , a , b , c , d , model.options.simf ));
        
        [ n , a , b , c , d ] = FindComponent( model.sv.X' , X' );
        Ksvx = Measures( n , a , b , c , d , model.options.simf );
    elseif strcmp(model.options.ker,'euclidean')
%             x2 = diag(1-pdist2(X',X','hamming'));
%             Ksvx = 1-pdist2(X',model.sv.X','hamming');
            
            [ n , a , b , c , d ] = FindComponent( X' , X' );
            x2 = diag(Measures( n , a , b , c , d , 'euclidean' ));

            [ n , a , b , c , d ] = FindComponent( model.sv.X' , X' );
            Ksvx = Measures( n , a , b , c , d , 'euclidean' );
    else
        x2 = diag(kernel( X, model.options.ker, model.options.arg));
        Ksvx = kernel( X, model.sv.X, model.options.ker, model.options.arg);
    end

    d = x2 - (2 * Ksvx * model.Alpha(:)) + (model.b * ones(num_data,1));

end