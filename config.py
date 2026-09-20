# config.py

SEED = 42
CORR_THRESHOLD = 0.85
BULK_QTY = 12
B2B_THRESHOLD = 0.5

FORCE_KEEP = [
    'Recency', 'MonthlyOrderRate', 'ReturnRate', 
    'QuarterConcentration', 'SKU_HHI'
]

CANDIDATES = [
    'Recency', 'InterPurchaseCV', 'ActiveSpanDays', 'MonthlyOrderRate',
    'AOV', 'SpendGini', 'PriceCV', 
    'NewSKURate', 'RepeatSKUFraction', 'BasketSizeCV',
    'BurstIndex', 'BulkLineRate', 'QuantityCV', 'MedianBasketQty', 'SKU_HHI',
    'MaxSpendShare', 'ReturnRate', 'ReturnValueRate',
    'SpendAcceleration', 'QuarterConcentration'
]

FAMILIES = {
    'Purchase Rhythm':   ['Recency', 'InterPurchaseCV', 'ActiveSpanDays', 'MonthlyOrderRate'],
    'Spending Shape':    ['AOV', 'SpendGini', 'PriceCV', 'MaxSpendShare', 'SpendAcceleration'],
    'Basket Behaviour':  ['NewSKURate', 'RepeatSKUFraction', 'BasketSizeCV'],
    'Volume & Bulk':     ['BurstIndex', 'BulkLineRate', 'QuantityCV', 'MedianBasketQty', 'SKU_HHI'],
    'Customer Lifecycle': ['ReturnRate', 'ReturnValueRate', 'QuarterConcentration']
}