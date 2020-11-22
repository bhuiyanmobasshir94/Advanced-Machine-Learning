## Welcome

Welcome to “How to win a data science competition” course! We are excited to have you in the class and we are looking forward to your contributions to the learning community.

Among all topics of data science, competitive data analysis is especially interesting. For an experienced specialist this is a great area to try his skills against other people and learn some new tricks; and for a novice this a good start to quickly and playfully learn basics of practical data science. For both, engaging in a competition is a good chance to expand the knowledge and get acquainted with new people.

But, despite that competitive data analysis is useful for both experts and novices, this particular course is designed for advanced students: while we teach ABC of data science competitions, we assume you are already familiar with machine learning ABC. If you only begin your journey in data science, we encourage you to check out other machine learning courses on Coursera. If you are familiar with python, sklearn and already have some experience in training machine learning models, we hope this course will prove useful to you.

As in any competitive field, you need to work very hard to get a prize in a competition. But you also need to work efficiently -- you need to develop an intuition and be able to quickly determine how it is best to approach a given competition -- how to preprocess the data, extract features, how to set up the validation correctly and optimize the given metric. You should know the potential sources of data leakages, what parameters to tune in your favorite models, how to generate powerful features, how to ensemble the models. This is just an essential checklist, that every top data scientist should know. Beyond that, every new competition requires you to craft new techniques and approaches, and as you will go through them, your experience and expertise will grow.

In this course, you will learn the essential techniques, mentioned above. We will also illustrate creativity in solutions with a detailed breakdown of our solutions in a number of competitions, in which the instructors of this course got top places -- a truly unique opportunity to see the detailed explanations of the winning solutions.

Everyone who works hard will learn a lot. But because this is a competition, only the most creative, dedicated and experienced people will be able to outperform others and get to the very top. But still, the obvious truth is, the more time you spend on a competition, the more you will be able to learn.

This course is delivered to you by the people who spent much time competing and mostly learned everything by trials and errors. We started our journey in competitive data science a while back, and were brought together by "Machine Learning Training" meetups in Moscow, which were organized by our common friend in Yandex. We successfully participated in wide range of competitions on Kaggle and other platforms since then, competing, learning and sharing knowledge with many other people. This was really inspiring time when we learned that through collaboration and hard work you can accomplish so much.

We personally are very grateful to Alexander Djakonov, a former top1 at Kaggle, who taught a course on data science and shared a lot of his secrets. His course was a truly exciting journey for us and now this is our time to share our knowledge. Also, we want to specially thank Stanislav Semenov for consulting about study plan, lectures and assignments.

Good luck as you get started and we hope you enjoy the course!

#### Week 1 overview

Welcome to the first week of the "How to Win a Data Science Competition" course! This is a short summary of what you will learn.

Mikhail Trofimov will introduce you to competitive data science. You will learn about competitions' mechanics, the difference between competitions and a real-life data science, overview hardware and software that people usually use in competitions. We will also briefly recap major ML models frequently used in competitions.
Alexander Guschin will summarize approaches to work with features: preprocessing, generation and extraction. We will see, that the choice of the machine learning model impacts both preprocessing we apply to the features and our approach to generation of new ones. We will also discuss feature extraction from text with Bag Of Words and Word2vec, and feature extraction from images with Convolution Neural Networks.
Let's go ahead and get started!

1. [Random Forest Algorithm](https://www.datasciencecentral.com/profiles/blogs/random-forests-explained-intuitively)
2. [Gradient Boosting explained](http://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)
3. [Reconstructing pictures with machine learning](http://arogozhnikov.github.io/2016/02/09/DrawingPictureWithML.html)
4. [Introduction to k-Nearest Neighbors: A powerful Machine Learning Algorithm](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)
5. [vowpal_wabbit](https://github.com/VowpalWabbit/vowpal_wabbit)
6. [xgboost](https://github.com/dmlc/xgboost)
7. [LightGBM](https://github.com/Microsoft/LightGBM)
8. [Interactive demo of simple feed-forward Neural Net](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.07950&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
9. [Example from sklearn with different decision surfaces](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
10. [Arbitrary order factorization machines](https://github.com/geffy/tffm)
11. [Basic SciPy stack (ipython, numpy, pandas, matplotlib)](https://www.scipy.org/)
12. [Stand-alone python tSNE package](https://github.com/danielfrg/tsne)
13. Libraries to work with sparse CTR-like data: [LibFM](http://www.libfm.org/), [LibFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)
14. [Blog "datas-frame" (contains posts about effective Pandas usage)](https://tomaugspurger.github.io/)
15. [Preprocessing in Sklearn](http://scikit-learn.org/stable/modules/preprocessing.html)
16. [Andrew NG about gradient descent and feature scaling](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling)
17. [Feature Scaling and the effect of standardization for machine learning algorithms](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)
18. [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
19. [Discussion of feature engineering on Quora](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)
20. [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
21. [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)
22. [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)
23. [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)
24. [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
25. [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)
26. [NLTK](http://www.nltk.org/)
27. [TextBlob](https://github.com/sloria/TextBlob)
28. [Using pretrained models in Keras](https://keras.io/applications/)
29. [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)
30. [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
31. [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)