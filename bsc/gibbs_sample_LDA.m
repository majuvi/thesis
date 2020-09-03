function [ q_dist, q_sample ] = gibbs_sample_LDA( document, topics, words, iterations, alpha, lambda )
    %Gibbs_sample_LDA Produces topic distribution for given documents using
    %Gibbs sampling for the Latent Dirichlet Allocation (LDA) model.
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
    %   q: NxV topic allocations
    %   c: KxVxN I(q(ij) == k, y(ij) == v)
    %   q_k: P(q(i) == k | .)
    %
    % Returns:
    %   q_dist: topic distributions cell array
    %   q_sample: word topic allocations matrix

    documents = size(document,2);
    words_per_doc = [];
    for j=1:documents
        words_per_doc(j) = size(document{j},1);
    end
    
    q = {};
    q_sample = {};
    for i = 1:documents
        q{i} = floor(rand([words_per_doc(i), 1])*topics+1);
        q_sample{i} = zeros([words_per_doc(i), 1]);
    end

    c = [];
    for i=1:documents
        kv = zeros(topics, words);
        for l=1:words_per_doc(i)
            kv(q{i}(l), document{i}(l))= kv(q{i}(l), document{i}(l)) + 1;
        end
        c(:,:,i) = kv;
    end
    
    %c_kvi
    c_ki = sum(c,2);
    c_kv = sum(c,3);
    c_k = sum(c_kv,2);
                
    n_samples = 10;
    offset = mod(iterations,n_samples);
    block = floor(iterations/n_samples);

    fprintf('starting...\n');
    for iter = 1:iterations
        for i = 1:documents
            for l=1:words_per_doc(i)

                q_il = q{i}(l);
                y_il = document{i}(l);

                %same as c(q_il,y_il,i) = c(q_il,y_il,i) - 1;
                c_ki(q_il,1,i) = c_ki(q_il,1,i) - 1;
                c_kv(q_il,y_il,1) = c_kv(q_il,y_il,1) - 1;
                c_k(q_il) = c_k(q_il) - 1;

                p_q = zeros(1,topics);
                for k=1:topics
                    p_q(k) = (c_kv(k,y_il)+lambda) / (c_k(k,1)+words*lambda) * (c_ki(k,1,i)+alpha) / ( words_per_doc(i) + topics*alpha);
                end

                k = randsample(topics, 1, true, p_q);

                %same as c(k,y_il,i) = c(k,y_il,i) + 1;
                c_ki(k,1,i) = c_ki(k,1,i) + 1;
                c_kv(k,y_il,1) = c_kv(k,y_il,1) + 1;
                c_k(k) = c_k(k) + 1;
                
                q{i}(l) = k;
            end
        end
        if (mod(iter-offset,block) == 0)
            fprintf('%d/%d\n',iter,iterations);
            for j = 1:documents
                q_sample{j} = q_sample{j} + q{j};
            end
        end
    end

    for j = 1:documents
        q_sample{j} = round(q_sample{j}./n_samples);
    end
    
    q_dist = zeros([documents,topics]);
    for i = 1:documents
        for j = 1:topics
            s = size(find(q_sample{i}==j));
            q_dist(i,j) = s(1,1);
        end
        q_dist(i,:) = q_dist(i,:)./sum(q_dist(i,:));
    end
    
    fprintf('done.\n');
end

