from utils import get_all_h_param_comb

def test_hparam_combinations():
    gammas = [0.1, 0.1, 0.005, 0.123, 0.879, 0.009]
    cparams = [1, 10, 100]

    comb = get_all_h_param_comb(gammas, cparams) 

    expected_p1 = (0.1, 1)

    assert len(comb) == len(gammas) * len(cparams)
    
    assert expected_p1 in comb 