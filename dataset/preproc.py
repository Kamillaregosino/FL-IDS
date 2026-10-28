import pandas as pd

# from ann_visualizer.visualize import ann_viz


# https://www.kaggle.com/code/keysa15/neural-network-based-intrusion-detection-system

# df = pd.read_csv('kddcup.data_10_percent.gz', header=None) # (494.021, 42)
df = pd.read_csv('kddcup.data.gz', header=None) # (4.898.431, 42)
cols = pd.read_csv('kddcup.names', header=None)

print(df.shape)
with open('training_attack_types', 'r') as f:
    attack_types = f.read()
    print(attack_types)


## Creating a dictionary of attack types
types = dict()
types['normal'] = 'normal'
for line in attack_types.split("\n"):
    if line:
        attack, description = line.split(" ", 1)
        types[attack] = description


if cols[0][0] == 'back':
    cols = cols.drop(cols.index[0])
    cols.reset_index(drop=True, inplace=True)

cols = cols.dropna(axis=1)
cols.head()

cols[[0,1]] = cols[0].str.split(':',expand = True)

names = cols[0].tolist()
names.append('label')
df.columns = names

df['Attack Type'] = df['label'].apply(lambda x: types[x[:-1]])
df.head()

# df.drop('protocol_type', axis=1, inplace=True)
# df.drop('num_root', axis=1, inplace=True)
df.drop('duration', axis=1, inplace=True)
# df.drop('protocol_type', axis=1, inplace=True)
# df.drop('flag', axis=1, inplace=True)
# df.drop('src_bytes', axis=1, inplace=True)
# df.drop('dst_bytes', axis=1, inplace=True)
# df.drop('land', axis=1, inplace=True)
# df.drop('wrong_fragment', axis=1, inplace=True)

# df.drop('urgent', axis=1, inplace=True)

df.drop('count', axis=1, inplace=True)
df.drop('srv_count', axis=1, inplace=True)
df.drop('serror_rate', axis=1, inplace=True)
df.drop('rerror_rate', axis=1, inplace=True)
df.drop('same_srv_rate', axis=1, inplace=True)
df.drop('diff_srv_rate', axis=1, inplace=True)

df.drop('srv_diff_host_rate', axis=1, inplace=True)
df.drop('dst_host_count', axis=1, inplace=True)
df.drop('dst_host_srv_count', axis=1, inplace=True)
df.drop('dst_host_diff_srv_rate', axis=1, inplace=True)
df.drop('dst_host_same_src_port_rate', axis=1, inplace=True)
df.drop('dst_host_srv_diff_host_rate', axis=1, inplace=True)
df.drop('label', axis=1, inplace=True)
df.drop('service', axis=1, inplace=True)


# print(df.shape)

df_novo = pd.concat([
    df[df['Attack Type'] == 'dos'].sample(n=40000, random_state=1),
    df[df['Attack Type'] == 'normal'].sample(n=40000, random_state=1),
    df[df['Attack Type'] == 'probe'].sample(n=40000, random_state=1),
    df[df['Attack Type'] == 'r2l'].sample(frac=1),
    df[df['Attack Type'] == 'u2r'].sample(frac=1)]
).sample(frac=1, random_state=1)


# df.to_csv('__processed_kdd.csv', index=False)
df_novo.to_csv('__processed_kdd.csv', index=False)