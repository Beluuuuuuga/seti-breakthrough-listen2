import pandas as pd

train = pd.read_csv("../input/seti-breakthrough-listen/train_labels.csv")
# test = pd.read_csv('../input/seti-breakthrough-listen/sample_submission.csv')
# test = pd.read_csv('../output/ef4_space6_512_mixup.csv') # train 005
# test = pd.read_csv('../output/ef4_space6_640_mixup.csv') # train 006
# test = pd.read_csv('../output/submit_2021_05_31_v1.csv') # best
# test = pd.read_csv('../output/seti_ensemble_ver0613_ver1_by_yyama_best.csv') # best
# test = pd.read_csv('../output/seti_ensemble_ver0613_ver1_by_yyama_best.csv') # best
test = pd.read_csv("../input/seti-breakthrough-listen/outputs/sugawara_lb_0790.csv")


def get_train_file_path(image_id):
    return "./input/seti-breakthrough-listen/train/{}/{}.npy".format(
        image_id[0], image_id
    )


def get_test_file_path(image_id):
    return "./input/seti-breakthrough-listen/test/{}/{}.npy".format(
        image_id[0], image_id
    )


train["file_path"] = train["id"].apply(get_train_file_path)
test["file_path"] = test["id"].apply(get_test_file_path)

print(train.head())
print(test.head())

# above_095 = test[test["target"] > 0.95]
above_090 = test[test["target"] > 0.90]
above_080 = test[test["target"] > 0.80]
above_070 = test[test["target"] > 0.70]
above_065 = test[test["target"] > 0.65]
above_060 = test[test["target"] > 0.60]
above_050 = test[test["target"] > 0.50]
below_040 = test[test["target"] < 0.4]
below_030 = test[test["target"] < 0.3]
below_020 = test[test["target"] < 0.2]
below_010 = test[test["target"] < 0.1]
print("above 0.90", round(len(above_090) / len(test) * 100, 2))
print("above 0.80", round(len(above_080) / len(test) * 100, 2))
print("above 0.70", round(len(above_070) / len(test) * 100, 2))
print("above 0.65", round(len(above_065) / len(test) * 100, 2))  # 3.84% v3
print("above 0.60", round(len(above_060) / len(test) * 100, 2))  # 8.91% v2
print("above 0.50", round(len(above_050) / len(test) * 100, 2))
print("below 0.40", round(len(below_040) / len(test) * 100, 2))
print("below 0.30", round(len(below_030) / len(test) * 100, 2))  # 73.21% v2
print("below 0.20", round(len(below_020) / len(test) * 100, 2))  # 7.09% v3
print("below 0.10", round(len(below_010) / len(test) * 100, 2))


# exit()

# df_v = pd.concat([train, below_005, above_090], axis=0) # v1
# df_v = pd.concat([train, below_003, above_095], axis=0) # v2
# df_v = pd.concat([below_030, above_060], axis=0)  # v1
# df_v = pd.concat([below_030, above_060], axis=0)  # v2
df_v = pd.concat([below_020, above_065], axis=0)  # v2
# # df_v = df_v.reset_index()


def preprocessing_target(row):
    if row.target < 0.20:
        return 0
    if row.target > 0.65:
        return 1


df_v["target"] = df_v.apply(preprocessing_target, axis=1)
# df_v.to_csv("../input/train_labels_pseudo_v2.csv", index=False)
df_v.to_csv(
    # "../input/seti-breakthrough-listen/seti_sugawara_lb_0790_ver2_above_060_below_030.csv", # v2
    "../input/seti-breakthrough-listen/seti_sugawara_lb_0790_ver3_above_065_below_020.csv",  # v3
    index=False,
)
