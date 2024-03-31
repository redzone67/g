import numpy as np
import pandas as pd
from scipy.signal import hilbert
from datetime import datetime 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def make_hilvert(arg_data):
    analytic_s = hilbert(arg_data, axis=0)
    C_ = analytic_s.conj().T.dot(analytic_s)/arg_data.shape[0]
    arg_evals, arg_evecs = np.linalg.eig(C_)
    idx = np.real(arg_evals).argsort()[::-1]
    arg_evals = (np.real(arg_evals))[idx]
    arg_evecs = arg_evecs[:,idx]
    arg_evecs = pd.DataFrame(arg_evecs)
    arg_evecs.index=list(arg_data.columns)

    alpha_j_t = analytic_s.dot(np.conj(arg_evecs).T)
    arg_mode_signal = pd.DataFrame(np.abs(alpha_j_t))
    arg_mode_signal.index = list(arg_data.index)

    mode_signal2 = pd.DataFrame(np.abs(alpha_j_t)**2)
    mode_signal2.index = list(arg_data.index)
    arg_intensity = mode_signal2.div(mode_signal2.sum(axis=1),axis=0)
        
    return arg_evals,arg_evecs,arg_mode_signal,arg_intensity


def rotational_shuffle(arg_data):
    """
    データを変数ごとに回転的にシャッフルします。
    data: 時系列データ（時間が行、変数が列）
    """
    N, num_vars = arg_data.shape  # Nは時系列の長さ、num_varsは変数の数
    shuffled_data = pd.DataFrame(np.empty_like(arg_data),columns=list(arg_data.columns))
    
    for var in list(arg_data.columns):
        tau = np.random.randint(0, N-2)  # 変数ごとに異なるランダムオフセットtauを生成
        shuffled_data.loc[:, var] = np.roll(arg_data.loc[:, var], -tau)
    
    return shuffled_data


def process_shuffled_data(arg_data):
    shuffled = rotational_shuffle(arg_data)
    shuffled_eigenvals,_,_,_ = make_hilvert(shuffled)
    return np.sort(np.real(shuffled_eigenvals))[::-1]


##### 並列処理
def make_rss_multiprocess(arg_sim_num, arg_input_data, arg_eigenvals):
    elapse = datetime.now()
    # プロセスごとに処理するデータのリストを作成
    num_processes = multiprocessing.cpu_count()
    
    # arg_input_dataをリストに格納して渡す
    args = [arg_input_data]*arg_sim_num
    
    # タスクを均等に分配
    #chunk_size = max(1, int(arg_sim_num / num_processes))
    #chunks = [args[i:i + chunk_size] for i in range(0, len(args), chunk_size)]
    #print('チャンクサイズ: ',chunk_size)
    #print('チャンクの数:',[len(s) for s in chunks])

    # ProcessPoolExecutorを使用して並列処理
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        arg_results = list(executor.map(process_shuffled_data, args))

    # 結果のリストを平坦化
    #arg_results = [item for sublist in arg_results for item in sublist]

    eigenval_distribution = pd.DataFrame(arg_results)
    eigenval_mean = pd.DataFrame(eigenval_distribution.mean()).T
    eigenval_std = pd.DataFrame(eigenval_distribution.std()).T
    thresholds = (eigenval_mean + 3 * eigenval_std).T
    thresholds.columns = ['RRS']

    Q = arg_input_data.shape[0] / (2 * arg_input_data.shape[1])
    lambda_upp = (1 + np.sqrt(1/Q))**2

    arg_eigenvals = pd.DataFrame(arg_eigenvals)
    arg_eigenvals.columns = ['actual']
    arg_chk = pd.concat([arg_eigenvals,thresholds],axis=1)
    arg_chk['RTM'] = lambda_upp
    arg_chk.loc[:,'RRS_chk']=(arg_chk.loc[:,'actual']>arg_chk.loc[:,'RRS'])
    arg_chk.loc[:,'RMT_chk']=(arg_chk.loc[:,'actual']>arg_chk.loc[:,'RRS'])
    arg_chk = arg_chk[(arg_chk['RRS_chk']==True)|(arg_chk['RMT_chk']==True)]

    print('並列処理時間: ',datetime.now()-elapse)
    return arg_chk

##### ここまで
    

def make_rss(arg_sim_num,arg_input_data,arg_eigenvals):
    elapse = datetime.now()
    arg_results = []
    for i in range(arg_sim_num):
        arg_results += [process_shuffled_data(arg_input_data)]

    eigenval_distribution = pd.DataFrame(arg_results)
    eigenval_mean = pd.DataFrame(eigenval_distribution.mean()).T
    eigenval_std = pd.DataFrame(eigenval_distribution.std()).T
    thresholds = (eigenval_mean + 3 * eigenval_std).T
    thresholds.columns = ['RRS']

    Q = arg_input_data.shape[0] / (2 * arg_input_data.shape[1])
    lambda_upp = (1 + np.sqrt(1/Q))**2

    arg_eigenvals = pd.DataFrame(arg_eigenvals)
    arg_eigenvals.columns = ['actual']
    arg_chk = pd.concat([arg_eigenvals,thresholds],axis=1)
    arg_chk['RTM'] = lambda_upp
    arg_chk.loc[:,'RRS_chk']=(arg_chk.loc[:,'actual']>arg_chk.loc[:,'RRS'])
    arg_chk.loc[:,'RMT_chk']=(arg_chk.loc[:,'actual']>arg_chk.loc[:,'RRS'])
    arg_chk = arg_chk[(arg_chk['RRS_chk']==True)|(arg_chk['RMT_chk']==True)]
    print('ループ処理時間: ',datetime.now()-elapse)

    return arg_chk


def make_sig(arg_sim_num,arg_data):
    rng = np.random.default_rng() #　平均ゼロ、標準偏差１の乱数
    dataX = arg_data.copy()
    for i in range(arg_sim_num):
        dataX.loc[:,'rand'] = rng.standard_normal(dataX.shape[0])
        _,evecs,_,_ = make_hilvert(dataX)
        evecs = evecs.iloc[-1,:-1].values
        if i == 0:
            arg_sig = evecs
        else:
            arg_sig = np.vstack([arg_sig,evecs])

    arg_sig = np.abs(arg_sig)
    arg_sig_mean = arg_sig.mean(axis=0)
    arg_sig_std = arg_sig.std(axis=0)

    return arg_sig_mean, arg_sig_std


def make_phase(arg_eigenvecs,arg_var):
    variable_names = list(arg_eigenvecs.index)
    new_phases = []
    new_mags = []
    for i in range(arg_eigenvecs.shape[1]):  # 列（固有ベクトル）ごとにループ
        arg_vec = arg_eigenvecs.iloc[:, i]  # i番目の固有ベクトル
        arg_phases = np.angle(arg_vec)  # 位相を計算
        arg_magnitudes = np.abs(arg_vec)  # 絶対値を計算

        # 基準成分の選択（例えば、0番目の成分）と相対的な位相の調整
        pos = variable_names.index(arg_var)
        reference_phase = arg_phases[pos]
        relative_phases = arg_phases - reference_phase
        # [-π, π]の範囲に調整
        relative_phases = np.mod(relative_phases + np.pi, 2 * np.pi) - np.pi
        
        # 反相関の閾値を定義
        for j, phase in enumerate(relative_phases):
            # 反相関成分の判定と処理
            if np.isclose(phase, np.pi, atol=1e-2):  # atolは許容誤差
                relative_phases[j] = (phase + np.pi) % (2 * np.pi) - np.pi
                arg_magnitudes[j] *= -1
        new_phases += [relative_phases/np.pi]
        new_mags += [arg_magnitudes]

    new_phases = pd.DataFrame(new_phases).T
    new_phases.index = variable_names
    new_mags = pd.DataFrame(new_mags).T

    ßreturn new_phases, new_mags

