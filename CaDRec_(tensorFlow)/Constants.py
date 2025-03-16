PAD = 0
Ks = [1, 5, 10, 20]
print('k:', Ks)

## Carl - Dataset Selection
## Carl - Modify this variable to change which dataset is being used by the program
DATASET = "douban-book"  # Yelp2018  Gowalla  Foursquare ml-1M douban-book
## Carl - Ablation options
ABLATIONs = {'w/oWeSch', 'w/oPopDe', 'w/oSA', 'w/oNorm', 'w/oUSpec', 'w/oHgcn', 'w/oDisen'}
ABLATION = 'Full'

## Carl - Number of users in each dataset
user_dict = {
    'ml-1M': 6038,
    'douban-book': 12859,
    'Gowalla': 18737,
    'Yelp2018': 31668,
    'Foursquare': 7642
}

## Carl - Number of items in each dataset
item_dict = {
    'ml-1M': 3533,
    'douban-book': 22294,
    'Gowalla': 32510,
    'Yelp2018': 38048,
    'Foursquare': 28483
}

ITEM_NUMBER = item_dict.get(DATASET)
USER_NUMBER = user_dict.get(DATASET)


print('Dataset:', DATASET, '#User:', USER_NUMBER, '#ITEM', ITEM_NUMBER)
print('ABLATION: ', ABLATION)

## Carl - Beta values for each dataset
beta_dict = {
    'ml-1M': 0.42,  # best 0.42,  # 0.5,  # 0.8  # 3629
    'douban-book': 0.07,  # best 0.07, 0.05,
    'Gowalla': 0.1,
    'Yelp2018': 0.25,  # best 0.25,
    'Foursquare': 0.5,  # best 0.5
}

## Carl - Eta values for each dataset
eta_dict = {
    'ml-1M': 0.06,  # best 0.42,  # 0.5,  # 0.8  # 3629
    'douban-book': 0.65,  # best 0.07, 0.05,
    'Gowalla': 0.1,
    'Yelp2018': 0.6,  # best 0.25,
    'Foursquare': 0.04,  # best 0.5
}

## Set beta value based on dataset and modify depending on ablation test
BETA_1 = beta_dict[DATASET]
if ABLATION == 'w/oPopDe' or ABLATION == 'w/oDisen': BETA_1 = 0

# During ablation study, in below cases, the popularity features will be over weighted, leading to over low accuracies.
if ABLATION == 'w/oSA' and DATASET == 'Yelp2018': BETA_1 = 0.01
if ABLATION == 'w/oSA' and DATASET == 'Yelp': BETA_1 = 0
if ABLATION == 'w/oHgcn' and DATASET == 'Gowalla': BETA_1 = 0
if ABLATION == 'OlyHGCN' and DATASET == 'Yelp2018': BETA_1 = 0.05

print('BETA_1', BETA_1)






