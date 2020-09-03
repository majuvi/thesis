%
% DEMO_NBC: demonstrates Naive Bayes Classifier (NBC)
%
%   => Generates documents as word vectors in a cell array
%   => Visualizes the documents as a bag-of-words bit image
%   => Samples topic and word distributions using Gibbs for NBC
%   => Visualizes the topic distribution as a bit image
%

% generate the documents
documents = 16; topics = 2; words = 5; words_per_doc = 15;
document = generate_documents( documents, topics, words, words_per_doc );

% visualize the documents
visualize = word_dist(document, words);
figure_distmatrix(visualize, 'Word distribution matrix', 'document', 'word');

% sample the topics
[q] = gibbs_sample_NBC(document, topics, words, 103, 1 ,1);

% visualize the topics
figure_distmatrix(q, 'Topic distribution matrix', 'document', 'topic');
