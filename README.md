Steps to run the system:

These are divided into two parts--- the preprocessing to create the dataset, and then the learning algorithm to process it.

Preprocessing requires multiple resources:
* Turkish KENET https://universaldependencies.org/treebanks/tr_kenet/index.html
* Turkish Unimorph
* The ELP dataset https://elexicon.wustl.edu/
* Subtlex https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus
* marry.py https://github.com/unimorph/ud-compatibility

1: Use ELP/Subtlex frequencies to compute reaction times
   /morph/rl/sh00_getData.py
   Function: calculate the log frequency and mean response time of the words from English lexicon project; Got English Lexicon Project data and regressed response times for mean visual lexical decision response latencies on log word frequency
   Input: /morph/rl/dataset/Items.csv
   Output:/morph/rl/dataset/data.csv

   /morph/rl/sh01_getELwordlist.py
   Function: Use English lexicon project as word list and get frequency count from SUBTLEXus corpus. Refit response times and regenerate graph (smaller bin figure) with 2 standard deviation lines within frequency bins (with 3-6 frequency bins). 
   Input: /morph/rl/dataset/data.csv
        /morph/rl/dataset/SUBTLEXusfrequencyabove1.csv
   Output:/morph/rl/dataset/elp_withsublex.csv

1b: Convert UD treebank to Unimorph using marry.py
   python marry.py convert --ud [path-to-turkish-ud].train.conllu -l tr

2: Get vocabulary of UD corpus
 - Get words and merge frequency data
 - Remove typos
 - Collapse to reasonable feature set/remove syncretic cells
 
  python rl/sh02b_getUD.py
     --project [path]/morph_rl 
     --ud_train [path]/tr_kenet-um-train.conllu
	 --pos_target NOUN
	 --pos_feat_style unimorph
     --local_frequency (for Turkish)
  >> rl/dataset/ud_UD_Turkish-Kenet.csv
  (note: dev, test in filenames refer to data extracted from conll dev/test split;
   this is not suitable as an inflection split)

3: Sample instances
- Split dataset into train/dev/test inflection split by lemma

  python rl/sh03b_splitByLemma.py
     --project [path]/morph_rl 
     --ud_dataframe rl/dataset/ud_UD_Turkish-Kenet.csv
  >> rl/dataset/ud_inflection_split_[train].csv

- Create Unimorph full-paradigm instances for testing

  python rl/sh03c_sampleUnimorph.py 
  		 --project [path]/morph_rl
		 --pos_target NOUN
		 --unimorph_data ~/tur/tur 
		 --ud_dataframe rl/dataset/ud_UD_Turkish-Kenet.csv
		 --ud_train rl/dataset/UD_Turkish-Kenet_inflection_split_train.csv

4: Assemble queries and responses for each item
    python rl/sh10b_sequential_dataset.py
     --project /users/PAS1268/osu8210/morph_rl 
	 --language UD_Turkish-Kenet
	 --split train/dev/test
    >> rl/dataset/query_df_[train].csv

- Plot and verify the temporal predictions

  python rl/sh10d_plot_predictions.py

Now we can run the learning system. This is a copy of one version of Wu et al (see the neural-transducer/README.md file for more cites). For some reason, the code is located in neural-transducer/example/transformer/src --- I am not sure why.

You will need a conda environment with:

* pytorch https://pytorch.org/get-started/locally/
* pynini https://github.com/kylebgorman/pynini

Check the settings in example/transformer/src/adaptive_q.py:

You can run the baseline by setting "n_sources: 0" and the model by running "n_sources: 1". This will run a fairly standard version of Wu but on the train/test files you just generated. You can switch the string aligner on and off with "baseline_use_aligner".

You should not need to change other settings.

With this environment active, run:

    python -u example/transformer/src/adaptive_q.py --project [path] --language UD_Turkish-Kenet --run [arbitrary name]

Once you have run the baseline for a couple of epochs, you can precompute and cache the alignments to speed up the main program (saving you time on the GPU server) using:

    python -u example/transformer/src/assemble_cache.py --project [path] --language UD_Turkish-Kenet --cache_section [0] --epoch 10 --split train

This is optional. If you're going to do it, you should run 100 jobs with --cache_section 0...99.

You can evaluate the learned policy using:

    python -u example/transformer/src/evaluate_policy.py --project [path] --language UD_Turkish-Kenet --epoch [20] --split [test] --run [same name as above] --product_inference product_only

("Product only" forces the system to discard the learned value prediction, which is what we did in the paper.)

You can evaluate the policy on Unimorph by adding the argument:

    --force_test query_unimorph_UD_Turkish-Kenet_test.csv

The model results go to:

checkpoints/UD-Turkish-Kenet-[run name]

The evaluator produces a textual report (report_[epoch]_[split]_product_only.txt) and a csv file (statistics_[epoch]_[split]_product_only.csv). You can run the programs:

    python -u example/transformer/src/analyzeFrequentStrs.py --stats [stats] --policy [optimal|predicted|stop|wait]
    python -u example/transformer/src/analyzeMemoryUsage.py --stats [stats] --policy [optimal|predicted|stop|wait]
    python -u example/transformer/src/analyzeStoppingPoints.py --stats [stats] --policy [optimal|predicted|stop|wait]

These obtain analysis results about what the program is actually doing (as shown in the data tables in the paper).