**Content-aware Neural Hashing for Cold-start Recommendation**

**Citation:** Casper Hansen, Christian Hansen, Jakob Grue Simonsen, Stephen Alstrup, and Christina Lioma. 2020. Content-aware Neural Hashing for Cold-start Recommendation. In _Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval_ (_SIGIR ’20_). Association for Computing Machinery, New York, NY, USA, 971–980. 

This link contains a downloadable version of all data+code used in the paper: [https://www.dropbox.com/s/cyxgbgxvigbdw2h/cacf_data_code.zip?dl=0](https://www.dropbox.com/s/cyxgbgxvigbdw2h/cacf_data_code.zip?dl=0)

The code is written for TensorFlow v. 1.12 and 1.14

## Datasets

We use datasets from Yelp and Amazon with the following naming scheme:

- amazon: amazon dataset for the standard (in-matrix) setting
- amacold: amazon dataset for the cold-start (out-of-matrix) setting
- amacold_10p: amazon dataset for the cold-start setting with 10% training data. We also have _20, _30, _40 versions for 20-40% training data.
- Similar naming scheme for yelp (yelp and yelcold).
## Generating hash codes
main.py is the main file of the project and can be called with a number of parameters, where the most important ones are:

 - \-\-bits: the number of bits in the hash code (16-64 is used in the paper)
 - \-\-dataset: name of the dataset (e.g., amacold)

While training, a pickle file is saved containing evaluation results and the hash codes. The hash codes can be gotten from this file like: 

    user_codes = filecontent[-3]
    item_codes = filecontent[-2]
