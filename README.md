/morph/rl/sh00_getData.py
Function: calculate the log frequency and mean response time of the words from English lexicon project; Got English Lexicon Project data and regressed response times for mean visual lexical decision response latencies on log word frequency
Input: /morph/rl/dataset/Items.csv
Output:/morph/rl/dataset/data.csv
Run code: /morph/rl/sh00_getData.sh


/morph/rl/sh01_getELwordlist.py
Function: Use English lexicon project as word list and get frequency count from SUBTLEXus corpus. Refit response times and regenerate graph (smaller bin figure) with 2 standard deviation lines within frequency bins (with 3-6 frequency bins). 
Input: /morph/rl/dataset/data.csv
        /morph/rl/dataset/SUBTLEXusfrequencyabove1.csv
Output:/morph/rl/dataset/elp_withsublex.csv
Run code: /morph/rl/sh01_getELwordlist.sh


/morph/rl/sh02_getUD.py
Function: Use universal dependency corpus as wordlist and use frequency count from other corpus. Begin to build inflection instances
Input: /morph/previous/english_train_freq.csv
        /morph/previous/english_test_freq.csv
        /morph/previous/english_dev_freq.csv
        /morph/rl/dataset/elp_withsublex.csv
Output:/morph/rl/dataset/ud_dev.csv
        /morph/rl/dataset/ud_test.csv
        /morph/rl/dataset/ud_train.csv
Run code: /morph/rl/sh00_getData.sh


/morph/rl/sh03_create_sql_tbl.py
Function: Use sql database to generate dataset, but I didn't use it to generate dataset at the end since earlier code works fine and is fast enough


/morph/rl/sh04_rt.py
Function: Use universal dependency corpus as wordlist and use frequency count from other corpus. Begin to build inflection instances
Input: /morph/previous/english_train_freq.csv
        /morph/previous/english_test_freq.csv
        /morph/previous/english_dev_freq.csv
        /morph/rl/dataset/elp_withsublex.csv
Output:/morph/rl/dataset/ud_dev.csv
        /morph/rl/dataset/ud_test.csv
        /morph/rl/dataset/ud_train.csv
Run code: /morph/rl/sh00_getData.sh

/morph/rl/sh05_make_elp.py
Function: get log fequency from sublex dataset for words in english lexicon project
Input: /morph/rl/dataset/data.csv
        /morph/rl/dataset/SUBTLEXusfrequencyabove1.csv
Output:/morph/rl/dataset/elp_withsublex.csv
Run code: /morph/rl/sh05_make_elp.sh

/morph/rl/sh06_delete_yes.py
Function: delete words with feature 'Typo': 'Yes', 'Abbr': 'Yes'
Input: /morph/rl/sq_output/query_df.csv
Output:/morph/rl/sq_output/query_df.csv
Run code: /morph/rl/sh06_delete_yes.sh


/morph/rl/sh07_make_transformer_dataset.py
Function: generate dataset that match with Wu's transformer model but these dataset does not have queries, we use these dataset to test wu's model
Input: /morph/rl/dataset/new_ud_train_filter.csv
        or /morph/rl/dataset/new_ud_dev_filter.csv
        or /morph/rl/dataset/new_ud_test_filter.csv
Output:/morph/rl/dataset/english-train_withrt.csv
        or /morph/rl/dataset/english-test_withrt.csv
        or /morph/rl/dataset/english-dev_withrt.csv
Run code: /morph/rl/sh07_make_transformer_dataset.sh





/morph/neural-transducer/sh04_create_newData_1107.py
Function: align current dataset english-dev generated from /morph/rl//morph/rl/sh04_rt.py with UD_English-GUM dataset, get the features from UD_English-GUM
Input: /morph/previous/UD_English-GUM/english_dev.csv
        /morph/rl/dataset/ud_dev.csv
        /morph/rl/dataset/ud_train.csv
        /morph/rl/dataset/ud_test.csv
Output:/morph/rl/dataset/new_ud_dev.csv
        /morph/rl/dataset/new_ud_train.csv
        /morph/rl/dataset/new_ud_test.csv
Run code: /morph/neural-transducer/sh04_create_newData_1107.sh

/morph/neural-transducer/sh02_wuFromUDSyncretism.py
Function: script to discard syncretic cells and overspecified features
Input: /morph/rl/dataset/new_ud_train.csv
        /morph/rl/dataset/new_ud_dev.csv
        /morph/rl/dataset/new_ud_test.csv
Output:/morph/rl/dataset/new_ud_dev_filter.csv
        /morph/rl/dataset/new_ud_test_filter.csv
        /morph/rl/dataset/new_ud_train_filter.csv
Run code: /morph/neural-transducer/sh02_wuFromUDSyncretism.sh

/morph/rl/sh10_sequencial_dataset.py
Function: generate dataset with queries 
Input: /morph/rl/dataset/new_ud_train_filter.csv
        or /morph/rl/dataset/new_ud_dev_filter.csv
        or /morph/rl/dataset/new_ud_test_filter.csv
Output:/morph/rl/sq_output/query_df_train.csv
        or /morph/rl/sq_output/query_df_test.csv
        or /morph/rl/sq_output/query_df_dev.csv
Run code: /morph/rl/sh10_sequencial_dataset.sh

/morph/neural-transducer/sh03_encodeInstancesWu.py
Function: create multi-source inputs/show input format
Input: /morph/rl/sq_output/query_df_train.csv
        or /morph/rl/sq_output/query_df_test.csv
        or /morph/rl/sq_output/query_df_dev.csv
Output:/morph/neural-transducer/encoded_df_train.csv
        or /morph/neural-transducer/encoded_df_test.csv
        or /morph/neural-transducer/encoded_df_dev.csv
Run code: /morph/neural-transducer/sh03_encodeInstancesWu.sh

/morph/rl/sh01_generate_dataset.py
Function: use the multi-source inputs to create dataset to train wu's model
Input: /morph/neural-transducer/encoded_df_dev.csv
        or /morph/neural-transducer/encoded_df_train.csv
        or /morph/neural-transducer/encoded_df_test.csv
Output:/morph/neural-transducer/dev_dataset.csv
        or /morph/neural-transducer/train_dataset.csv
        or /morph/neural-transducer/test_dataset.csv
Run code: /morph/rl/sh01_generate_dataset.sh

##change the name of files:
/morph/neural-transducer/dev_dataset.csv --> english-dev
/morph/neural-transducer/train_dataset.csv --> english-train-high 
put these two files into /morph/neural-transducer/data/conll2017/all/task1

change the name of file:
/morph/neural-transducer/test_dataset.csv --> english-uncovered-test
put it into /users/PAS2062/delijingyic/project/morph/neural-transducer/data/conll2017/answers/task1
run sh example/transformer/trm-sig17.sh english to train wu's model

/morph/neural-transducer/sh05_calculate_rewards.py
Function: calculate stop states' reward
Input: /morph/neural-transducer/english-uncovered-test.tsv
        /morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test.tsv
        /morph/rl/sq_output/query_df_test.csv
Output:/morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test_rt.tsv
Run code: /morph/neural-transducer/sh05_calculate_rewards.sh

/morph/neural-transducer/sh06_calculate_inter_rewards.py
Function: calculate stop states' reward
Input: /morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test_rt.tsv
Output:/morph/neural-transducer/checkpoints/transformer/tagtransformer/sigmorphon17-task1-dropout0.3/english-high-.decode.test_rt_intermediate.tsv
Run code: /morph/neural-transducer/sh06_calculate_inter_rewards.sh