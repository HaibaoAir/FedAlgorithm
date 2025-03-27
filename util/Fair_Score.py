def find_statistical_partiy_score(protected_attr, data, labels, predictions):
    """
    SPD公平性指标，关系到预测结果的正类和负类的比例是否与保护属性的值有关
    """
    protected_pos = 0
    protected_neg = 0
    non_protected_pos = 0
    non_protected_neg = 0
    
    saValue = 0
    for i in range(len(protected_attr)):
        if protected_attr[i] == saValue:
            if predictions[i] == 1:
                protected_pos += 1
            else:
                protected_neg += 1
        else:
            if predictions[i] == 1:
                non_protected_pos += 1
            else:
                non_protected_neg += 1
                
    protected_ratio = protected_pos / (protected_pos + protected_neg)
    non_protected_ratio = non_protected_pos / (non_protected_pos + non_protected_neg)
    
    statistical_partiy_score = non_protected_ratio - protected_ratio
    return statistical_partiy_score

def find_equal_oppotunity_score(protected_attr, data, labels, predictions):
    """
    EOD公平性指标，只关心在正类上的预测结果的差异
    """
    protected_pos = 0
    protected_neg = 0
    non_protected_pos = 0
    non_protected_neg = 0
    
    tp_protected = 0
    tn_protected = 0
    fp_protected = 0
    fn_protected = 0
    
    tp_non_protected = 0
    tn_non_protected = 0
    fp_non_protected = 0
    fn_non_protected = 0
    
    saValue = 0
    for i in range(len(protected_attr)):
        if protected_attr[i] == saValue:
            if predictions[i] == 1:
                protected_pos += 1
            else:
                protected_neg += 1
                
            if predictions[i] == 1 and labels[i] == 1:
                tp_protected += 1
            elif predictions[i] == 1 and labels[i] == 0:
                fp_protected += 1
            elif predictions[i] == 0 and labels[i] == 1:
                fn_protected += 1
            else:
                tn_protected += 1
                
        else:
            if predictions[i] == 1:
                non_protected_pos += 1
            else:
                non_protected_neg += 1
                
            if predictions[i] == 1 and labels[i] == 1:
                tp_non_protected += 1
            elif predictions[i] == 1 and labels[i] == 0:
                fp_non_protected += 1
            elif predictions[i] == 0 and labels[i] == 1:
                fn_non_protected += 1
            else:
                tn_non_protected += 1
                
    protected_ratio = tp_protected / protected_pos
    non_protected_ratio = tp_non_protected / non_protected_pos
    
    equal_oppotunity_score = non_protected_ratio - protected_ratio
    return equal_oppotunity_score

def find_equal_opportunity_score_central(protected_attr, labels, predictions):
    """
    EOD公平性指标，只关心在正类上的预测结果的差异
    """
    tp_protected = 0
    tn_protected = 0
    fp_protected = 0
    fn_protected = 0
    
    tp_non_protected = 0
    tn_non_protected = 0
    fp_non_protected = 0
    fn_non_protected = 0
    
    saValue = 0
    for i in range(len(protected_attr)):
        if protected_attr[i] == saValue:
            if labels[i] == 1:
                if predictions[i] == 1:
                    tp_protected += 1
                else:
                    fn_protected += 1
            else:
                if predictions[i] == 1:
                    fp_protected += 1
                else:
                    tn_protected += 1
                
        else:                
            if labels[i] == 1:
                if predictions[i] == 1:
                    tp_non_protected += 1
                else:
                    fn_non_protected += 1
            else:
                if predictions[i] == 1:
                    fp_non_protected += 1
                else:
                    tn_non_protected += 1
                
    protected_part = tp_protected / (tp_protected + fn_protected)
    non_protected_part = tp_non_protected / (tp_non_protected + fn_non_protected)
    
    equal_opportunity_score = non_protected_part - protected_part
    return equal_opportunity_score


def find_equal_opportunity_score_distributed(protected_attr, labels, predictions):
    """
    EOD公平性指标，只关心在正类上的预测结果的差异
    """
    tp_protected = 0
    tn_protected = 0
    fp_protected = 0
    fn_protected = 0
    
    tp_non_protected = 0
    tn_non_protected = 0
    fp_non_protected = 0
    fn_non_protected = 0
    
    saValue = 0
    for i in range(len(protected_attr)):
        if protected_attr[i] == saValue:
            if labels[i] == 1:
                if predictions[i] == 1:
                    tp_protected += 1
                else:
                    fn_protected += 1
            else:
                if predictions[i] == 1:
                    fp_protected += 1
                else:
                    tn_protected += 1
                
        else:                
            if labels[i] == 1:
                if predictions[i] == 1:
                    tp_non_protected += 1
                else:
                    fn_non_protected += 1
            else:
                if predictions[i] == 1:
                    fp_non_protected += 1
                else:
                    tn_non_protected += 1
                
    protected_part_1 = tp_protected / (tp_protected + fn_protected)
    protected_part_2 = (tp_protected + fn_protected) / len(protected_attr)

    non_protected_part_1 = tp_non_protected / (tp_non_protected + fn_non_protected)
    non_protected_part_2 = (tp_non_protected + fn_non_protected) / len(protected_attr)

    return protected_part_1, protected_part_2, non_protected_part_1, non_protected_part_2