from pathlib import Path
import pandas as pd
import json
import numpy as np

data_path = Path("/users/PAS2062/delijingyic/project/morph/dataset")
alter = {
    'Д': 'жд',
    'т': 'ц',
    'ст': 'щ',
    'к': 'ч',
    'г': 'ж',
    'х': 'ш',
    'в': 'вль',
    'п': 'пль',
    'жд':'Д',
    'ц': 'т',
    'щ': 'ст',
    'ч': 'к',
    'ж': 'г',
    'ш': 'х',
    'вль': 'в',
    'пль': 'п',

}
vowel = ['о', 'е']
data = pd.read_csv(
    data_path / "DeriNetRU-0.5_noInflection.tsv",
    delimiter='\t',
    dtype=str,
    names=["base", "derived", "POS", "org_base", "org_derived"],
)


def bottom_up_dp_lcs(str_a, str_b):
    """
    longest common substring of str_a and str_b
    """
    # print("str_a",type(str_a,),str_a)
    # print("str_b",type(str_b),str_b)

    if len(str_a) == 0 or len(str_b) == 0:
        return ""
    dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]

    max_len = 0
    lis_comm = []
    ind_a = 0
    ind_b = 0
    end_a = 0
    end_b = 0
    lcs_str = ""
    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max([max_len, dp[i][j]])
                if max_len == dp[i][j]:
                    lcs_str = str_a[i - max_len:i]
                    if lcs_str not in lis_comm:
                        ind_a = int(i - max_len)
                        ind_b = int(j - max_len)
                        end_a = int(i)
                        end_b = int(j)
                        lis_comm.append(lcs_str)
                    else:
                        break

            else:
                dp[i][j] = 0
    return lcs_str, ind_a, ind_b, end_a, end_b


data['base_prefix'] = ''
data['derived_prefix'] = ''
data['base_suffix'] = ''
data['derived_suffix'] = ''
for i, row in data['base'].iteritems():
    if data.loc[i, 'base'] == data.loc[i, 'derived']:
        continue
    else:
        # lis.append(i, [data.loc[i,'base'], data.loc[i,'derived']])
        # print("base",row)
        # print("derive",data.loc[i,'derived'])
        lcs_str, start_base, start_derive, end_base, end_derive = bottom_up_dp_lcs(
            data.loc[i, 'base'], data.loc[i, 'derived'])
        # print("lcs_str",lcs_str)
        # print("index_base",type(start_base),start_base)
        # print("index_derive",type(start_derive),start_derive)
        # print("end_base",type(end_base),end_base)
        # print("end_derive",type(end_derive),end_derive)

        if len(lcs_str) > 0:
            base = data.loc[i, 'base']
            der = data.loc[i, 'derived']
            prefix_base = base[:start_base]
            suffix_base = base[end_base:]
            prefix_derive = der[:start_derive]
            suffix_derive = der[end_derive:]

            if len(suffix_base) > 0:
                key = suffix_base[0]
                if key in vowel:
                    suffix_base = suffix_base[1:]
            else:
                suffix_base = base[end_base:]
            if len(suffix_derive) > 0:
                key2 = suffix_derive[0]
                if key2 in vowel:
                    suffix_derive = suffix_derive[1:]
            else:
                suffix_derive = der[end_derive:]

            if len(suffix_base) > 0 and len(suffix_derive) > 0:
                if suffix_base[0] in alter and alter[
                        suffix_base[0]] == suffix_derive[0]:
                    suffix_base = suffix_base[1:]
                    suffix_derive = suffix_derive[1:]

            post_delete_base = prefix_base + lcs_str + suffix_base
            psot_delete_derive = prefix_derive + lcs_str + suffix_derive
            lcs_str1, start_base1, start_derive1, end_base1, end_derive1 = bottom_up_dp_lcs(
                post_delete_base, psot_delete_derive)

            if len(lcs_str1) > 0:
                prefix_base = post_delete_base[:start_base1]
                suffix_base = post_delete_base[end_base1:]
                prefix_derive = psot_delete_derive[:start_derive1]
                suffix_derive = psot_delete_derive[end_derive1:]

            if len(prefix_base) == 0 and len(prefix_derive) == 0:
                derivation_process_prefix = ""
            else:
                derivation_process_prefix = prefix_base + '.' + prefix_derive
                # print(derivation_process_prefix)

            if len(suffix_base) == 0 and len(suffix_derive) == 0:
                derivation_process_suffix = ""
            else:
                derivation_process_suffix = suffix_base + '.' + suffix_derive
            derivation = derivation_process_prefix + ',' + derivation_process_suffix
            # print("prefix_base",prefix_base)
            # print("suffix_base",suffix_base)
            # print("prefix_derive",prefix_derive)
            # print("suffix_derive",suffix_derive)
            data.loc[i, 'base_prefix'] = prefix_base
            data.loc[i, 'base_suffix'] = suffix_base
            data.loc[i, 'derived_prefix'] = prefix_derive
            data.loc[i, 'derived_suffix'] = suffix_derive
            data.loc[i,
                     'derivation_process_prefix'] = derivation_process_prefix
            data.loc[i,
                     'derivation_process_suffix'] = derivation_process_suffix
            data.loc[i, 'derivation_prefix,deravation_suffix'] = derivation
'''
        if len(lcs_str)>0: 
            lis_base=data.loc[i,'base'].split(lcs_str)
            lis_derive=data.loc[i,'derived'].split(lcs_str)
            
            print("lis_base",lis_base)
            print("lis_derive",lis_derive)
        if len(lis_base) == 3:
            prefix_base = lis_base[0]
            suffix_base = lis_base[-1]
            print("prefix_base",prefix_base)
            print("suffix_base",suffix_base)
            input()
        elif len(lis_base) == 2:
            if lis_base[0]==lcs_str:
                prefix_base='0'
                suffix_base=lis_base[-1]
            else:
                prefix_base=lis_base[0]
                suffix_base='0'
        data.loc[i,'base_prefix']=prefix_base
        data.loc[i,'base_suffix']=suffix_base

        if len(lis_derive) == 3:
            prefix_derive = lis_derive[0]
            suffix_derive = lis_derive[-1]
        elif len(lis_derive) == 2:
            if lis_derive[0]==lcs_str:
                prefix_derive='0'
                suffix_derive=lis_derive[-1]
            else:
                prefix_derive=lis_derive[0]
                suffix_derive='0'
    '''
data2 = data
for i, row in data['base_prefix'].iteritems():
    if len(row) > 0:
        data2 = data[(data["base_prefix"].notnull())
                     & (data["base_prefix"] != "")]
for i, row in data['base_suffix'].iteritems():
    if len(row) > 0:
        data2 = data[(data["base_suffix"].notnull())
                     & (data["base_suffix"] != "")]
data2 = data.loc[:, [
    "org_base",
    "org_derived",
    "POS",
    "derivation_prefix,deravation_suffix",
]]
data2.to_csv(data_path / "DeriNetRU-0.5_base_with_derivation.tsv",
             sep='\t',
             index=False)
data3 = data.loc[:, [
    "org_base", "org_derived", "POS", "derivation_process_prefix",
    "derivation_process_suffix", "base", "derived"
]]

data3.to_csv(data_path / "DeriNetRU-0.5_all.tsv", sep='\t', index=False)

data = data.loc[:, [
    "org_base", "org_derived", "POS", "derivation_prefix,deravation_suffix",
    "derivation_process_suffix", "base", "derived"
]]

# data["derivation_process_prefix"].fillna("", inplace=True)
# data["derivation_process_suffix"].fillna("", inplace=True)
data["derivation_prefix,deravation_suffix"].fillna("", inplace=True)
# for i, row in data['derivation_process_prefix'].iteritems():
#     if row == np.nan:
#         data.loc[i,'derivation_process_prefix']=''
# for i, row in data['derivation_process_suffix'].iteritems():
#     if row == np.nan:
#         data.loc[i,'derivation_process_suffix']=''
dictionary = {}
for i, row in data['derivation_prefix,deravation_suffix'].iteritems():
    if row in dictionary:
        dictionary[row] += 1
    else:
        dictionary[row] = 1

print(data)
# exit(0)
data = data.loc[:, [
    "org_base", "org_derived", "POS", "derivation_process_suffix", "base",
    "derived"
]]
data.to_csv(data_path / "DeriNetRU-0.5_deriveProcess.tsv",
            sep='\t',
            index=False)

pd.DataFrame.from_dict(dictionary, orient="index").to_csv(
    data_path / "derivation_process_count.cvs", index=True)

print(len(dictionary))
