import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
import seaborn as _sns

def compare_distribution_by_number(data, column_name, labels = _np.arange(0,10)):
    
    for label in labels:
        label_data = data.loc[data['label'] == label, column_name]

        _sns.kdeplot(label_data, label = str(label))
    ax = _plt.gca()
    ax.legend(bbox_to_anchor=(1.2,1))
    ax.set_xlabel(column_name.replace('_', ' '))
    ax.set_ylabel('Freq')
    ax.set_title('Distribution of %s' % column_name.replace('_', ' '), y=1.05)