import pandas as pd
from simplet5 import SimpleT5

df_train = pd.read_csv('../data/trainingdata_v3/datasets/train_text_splitted750_entities.tsv', sep='\t')
df_train = df_train.rename(columns={"entities":"target_text", "text":"source_text"})
df_train = df_train[['source_text', 'target_text']]
df_train['source_text'] = "medications: " + df_train['source_text']
df_train.loc[pd.isna(df_train['target_text'])] = ' '

df_dev = pd.read_csv('../data/trainingdata_v3/datasets/train_text_splitted750_entities.tsv', sep='\t')
df_dev = df_dev.rename(columns={"entities":"target_text", "text":"source_text"})
df_dev = df_dev[['source_text', 'target_text']]
df_dev['source_text'] = "medications: " + df_dev['source_text']
df_dev.loc[pd.isna(df_dev['target_text'])] = ' '
# print(df_train)
model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="TheLongSentance/MIMIC-III-t5-large-v1")

model.train(
    train_df=df_train,
    eval_df=df_dev, 
    source_max_token_len=770, 
    target_max_token_len=250, 
    batch_size=2,
    max_epochs=7,
    use_gpu=True,
)
