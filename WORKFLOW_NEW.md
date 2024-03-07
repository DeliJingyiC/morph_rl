1: Use ELP/Subtlex frequencies to compute reaction times
(Steps 1 and 2 of README; output /morph/rl/dataset/elp_withsublex.csv)

2: Get vocabulary of UD corpus
- Get words and merge frequency data
-- /morph/rl/sh02_getUD.sh > /morph/rl/dataset/ud_train.csv
- Remove typos
- Collapse to reasonable feature set/remove syncretic cells
- Remove instances which connect the lemma to cells syncretic with itself

  python rl/sh02b_getUD.py
     --project /users/PAS1268/osu8210/morph_rl 
     --ud_train previous/UD_English-GUM/en_gum-ud-train.conllu previous/UD_English-EWT/en_ewt-ud-train.conllu
	 --pos_target VERB
  >> rl/dataset/ud_[names].csv
  (note: dev, test in filenames refer to data extracted from conll dev/test split;
   this is not suitable as an inflection split)

3: Sample instances
- Split dataset into train/dev/test by lemma

  python rl/sh03b_splitByLemma.py
     --project /users/PAS1268/osu8210/morph_rl 
     --ud_dataframe rl/dataset/ud_UD_English-GUM_UD_English-EWT.csv
  >> rl/dataset/ud_inflection_split_[train].csv

4: Assemble queries and responses for each item
    python rl/sh10b_sequential_dataset.py
     --project /users/PAS1268/osu8210/morph_rl 
    >> rl/dataset/query_df_[train].csv

5: Run Wu transformer
   (Currently, some manual work needs to be done here. Generation of synthetic data for
   transformer pretraining; pretraining transformer; start new run using previous run as parent.
   Some fussiness with setting epoch numbers for runs resuming from previous work.)

- For single-source
- For multi-source
   python rl/sh11b_encode_instances.py `pwd`
    --single_source
   >> neural-transducer/data/reinf_inst/ud_[single]_[train]

  cd neural-transducer; sbatch example/transformer/trm_reinf.sh ud_[single|multi]

6: Merge Wu output back into dataframe
    python rl/sh12b_merge_predictions.py 
      --project `pwd` 
      --transformer_output neural-transducer/checkpoints/ud_multi/ 
      --noncumulative
    >> rl/dataset/prediction_output_[dev|test].csv

7: Compute rewards
    python rl/sh13b_compute_rewards.py
      --project `pwd`
    >> rl/dataset/rewards_[dev|test].csv

8: Train Q-network
    python rl/sh14b_encode_reward_instances.py `pwd` 
        --synthetic_multitask
        --noncumulative
    >> neural-transducer/data/reinf_inst/ud_reward-[train|test]
	[some manual messing around takes place]
	sbatch example/transformer/trm_regress.sh ud_reward
	
9: Evaluate policy
   python rl/sh15b_evaluate_policy.py `pwd`
    --transformer_output neural-transducer/checkpoints/ud_[corpus]_reward
    --language [ud_...]