function [model] = svc( data, varargin )

    %==========================================================================
    % Support Vector Clustering Algorithms          
    %==========================================================================  
    %   
    % Note that my codes are based on Statistical Pattern Recognition Toolbox 
    % (stprtool). See Ref. 5.
    % It should be needed to install stprtool at first. 
    %
    % Input:
    %   data [struct] Training data:
    %       .X [dim x num_data] Training input vectors.
    %       .y [1 x num_data] Output cluster labels
    %  X [dim x num_data] Input data.
    %  options [struct] Control parameters:
    %   .ker [string] Kernel identifier (default 'linear'). See 'help kernel'.
    %   .arg [1 x nargs] Kernel arguments.
    %   .C [1x1] Regularization constant (default []); 
    %    If C = 0 (or C=1, default), it is hard margin, that is, 
    %       the model do not allow bounded support vectors
    %   .method [string] SVC cluster labeling Method
    %       'CG': Complete Graph Labeling Method. See Ref. 1 (default)
    %       'DD': Delaunay Diagram Labeling Method. See Ref. 4
    %       'MST': Minimum Spanning Tree Labeling Method. See Ref. 4
    %       'KNN': K-Nearest Neighbor Labeling Method. See Ref. 4
    %               >> options.k (number of K, default=4)
    %       'SEP-CG': SEP-based CG Labeling Method. See Ref. 2
    %       'E-SVC': Equilibrium vector based Labeling Method. See Ref. 3
    %          >> options.NofK (number of clusters, default = 0 (automatic determination) ) 
    %          --> We can control the number of clusters of SVC directly!
    %
    % Output:
    %  model [struct] from svdd.m
    %      .confusion_matrix:  (row) predicted cluster, (column) true cluster
    %      .cluster_labels [1 x num_data] predicted cluster labels
    %                --> ** NOTE: BSVs are classified as outliers (indexed by 0)
    %
    % Example
    %   load ring.mat;
    %   data.X=input';   % data.X: p x N input patterns
    %   options=struct('method','CG','ker','rbf','arg',0.5,'C',0.1);
    %   [model]=svc(data,options); 
    %   plotsvc(data,model);
    %
    %
    %
    % References  
    % 1. A.Ben-Hur,D.Horn,H.T.Siegelmann and V.Vapnik. Support Vector Clustring.
    %    Journal of Machine Learning Research 2, 125-137, 2001
    % 2. J. Lee and D. Lee, An Improved Cluster Labeling Method for Support 
    %    Vector Clustering. IEEE TPAMI 27-(3), 461- 464, 2005.
    % 3. J. Lee and D. Lee, Dynamic Characterization of Cluster Structures for 
    %    Robust and Inductive Support Vector Clustering, IEEE TPAMI 28-(11), 
    %    1869-1874, 2006.
    % 4. J. Yang, V. Estivill-Castro, S.K. Chalup, Support Vector Clustering 
    %    Through Proximity Graph Modelling, Proc. 9th Int??l Conf. Neural 
    %    Information Processing, 898-903, 2002.
    % 5. Stprtool: http://cmp.felk.cvut.cz/cmp/software/stprtool/
    %
    %==========================================================================
    % January 13, 2009
    % Implemented by Daewon Lee
    % WWW: http://sites.google.com/site/daewonlee/
    %==========================================================================

    %% Initialization
    if ~isfield(data,{'y'})
        data.y = ones(size(data.X,1),1);
    end
    
    data.X = data.X';
    data.y = data.y';
    
    method  = find(strcmp('method', varargin));
    if ~isempty(method)
        method = varargin{method+1};
    else
        method = 'CG';
    end
    
    ker  = find(strcmp('ker', varargin));
    if ~isempty(ker)
        ker = varargin{ker+1};
    else
        ker = 'rbf';
    end
    
    simf  = find(strcmp('simf', varargin));
    if ~isempty(simf)
        simf = varargin{simf+1};
    else
        simf = 'tanimoto';
    end
    
    arg  = find(strcmp('arg', varargin));
    if ~isempty(arg)
        arg = varargin{arg+1};
    else
        arg = 1;
    end
    
    C  = find(strcmp('C', varargin));
    if ~isempty(C)
        C = varargin{C+1};
    else
        C = 1;
    end
    
    options = struct('method',method,'ker',ker,'arg',arg,'C',C,'simf',simf);

    %% Optimzing SVDD model
    %disp('Step 1: Optimizing the SVDD dual problem.....');
    model = svdd(data.X,options);

    %% Cluster Labeling
    %disp(strcat(['Step 2: Labeling cluster index by using ',options.method,'....']));
    switch options.method
        case 'CG'        
            adjacent = FindAdjMatrix(data.X(:,model.inside_ind),model);
            % Finds the cluster assignment of each data point 
            clusters = FindConnectedComponents(adjacent);
            model.cluster_labels = zeros(1,length(data.y));
            model.cluster_labels(model.inside_ind) = double(clusters);        
        case 'DD'
            [result] = plot_dd(data.X(:,model.inside_ind)');        
            adjacent = FindAdjMatrix_proximity(data.X(:,model.inside_ind),result,model);
            % Finds the cluster assignment of each data point 
            clusters = FindConnectedComponents(adjacent);
            model.cluster_labels = zeros(1,length(data.y));
            model.cluster_labels(model.inside_ind) = double(clusters);    
         case 'MST'
            [result] = plot_mst(data.X(:,model.inside_ind)');
            adjacent = FindAdjMatrix_proximity(data.X(:,model.inside_ind),result,model);
            % Finds the cluster assignment of each data point 
            clusters = FindConnectedComponents(adjacent);
            model.cluster_labels = zeros(1,length(data.y));
            model.cluster_labels(model.inside_ind)=  double(clusters);    
        case 'KNN'
            [result] = knn_svc(options.k,data.X(:,model.inside_ind));
            adjacent = FindAdjMatrix_proximity(data.X(:,model.inside_ind),result,model);
            % Finds the cluster assignment of each data point 
            clusters =  FindConnectedComponents(adjacent);        
            model.cluster_labels = zeros(1,length(data.y));
            model.cluster_labels(model.inside_ind) = double(clusters);  

        case 'SEP-CG'
            % find stable equilibrium points
            [rep_locals,locals,local_val,match_local] = FindLocal(data.X',model);
            model.local = locals';    % dim x N_local
            %small_n=size(locals,1);
            % calculates the adjacent matrix
            [adjacent] = FindAdjMatrix(model.local,model);
            % Finds the cluster assignment of each data point 
            local_clusters_assignments = FindConnectedComponents(adjacent);
            model.cluster_labels = local_clusters_assignments(match_local)';
        case 'E-SVC'
            if ~isfield(model.options,{'epsilon'})
                model.options.epsilon=0.05;
            end
            if ~isfield(model.options,{'NofK'})
                model.options.NofK=0;
            end        
           [model] = esvcLabel(data,model);
    end
    %disp('Finished SVC clustering!');
end
