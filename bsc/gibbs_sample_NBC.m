function [ q_sample ] = gibbs_sample_NBC( document, topics, words, iterations, alpha, lambda )
    %Gibbs_sample_NBC Produces topic allocation for given documents using
    %Gibbs sampling for the Naive Bayes Classifier (NBC) model.
    %
    % Arguments:
    %   document: documents as cell array of vectors of integers
    %   topics: number of topics to allocate
    %   words: number of words in dictionary
    %   iterations: iterations to run
    %   alpha: prior hyperparameter for the topic distribution
    %   lambda: prior hyperparameter for the word distribution
    %
    % Internal variables:
    %   q: N topic allocations
    %   c: KxVxN I(q(ij) == k, y(ij) == v)
    %   q_k: P(q(i) == k | .)
    %
    % Returns:
    %   q_sample: document topic allocations matrix

    documents = size(document,2);
    words_per_doc = [];
    for j=1:documents
        words_per_doc(j) = size(document{j},1);
    end
    
    q = floor(rand([documents, 1])*topics+1);
    q_sample = zeros([documents, 1]);
    
    b = zeros([topics,words]);
    for i = 1:topics
        b(i,:) = sample_dirichlet(lambda*ones([1, words]));
    end

    c = [];
    for i=1:documents
        kv = zeros(topics, words);
        for l=1:words_per_doc(i)
            kv(q(i), document{i}(l))= kv(q(i), document{i}(l)) + 1;
        end
        c(:,:,i) = kv;
    end

    n_samples = 10;
    offset = mod(iterations,n_samples);
    block = floor(iterations/n_samples);

    fprintf('starting...\n');
    for iter = 1:iterations
        for i = 1:documents
            
            q_i = q(i);
            
            %c_kvi
            c_vi = sum(c,1);
            c_kv = sum(c,3);
            c_k = sum((sum(c,2)~=0),3);
            
            p_q = zeros(1,topics);
            for k=1:topics
                p_q(k) = (alpha+c_k(k)-1)/(alpha*topics+documents-1);
                B = 1;
                for v = 1:words
                    B = B*(b(k,v)^(lambda+c_vi(1,v,i)-1));
                end
                p_q(k) = p_q(k)*B;
            end
            
            k = randsample(topics, 1, true, p_q);

            old = c(q_i,:,i);
            c(q_i,:,i) = zeros([1,words]);
            c(k,:,i) = old;
            
            q(i) = k;
        end
        
        c_kv = sum(c,3);
        for i = 1:topics
            b(i,:) = sample_dirichlet(c_kv(i,:)+lambda);
        end
        
        if (mod(iter-offset,block) == 0)
            fprintf('%d/%d\n',iter,iterations);
            q_sample = q_sample + q;
        end
    end

    q_sample = round(q_sample./n_samples);
    
    fprintf('done.\n'); 
end

