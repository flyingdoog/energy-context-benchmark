"""Query-time EPFL arrays shared by the native sequence reference (NumPy only)."""
import numpy as np

CONDITIONS=("power","power_state","full")
HISTORY_NAMES=("soc","total_capacity","remaining_capacity","bulk_capacity",
               "delivered_energy_wh","setpoint_w","requested_w","controlled")
STATIC_NAMES=("hour_sin","hour_cos","weekday","month")


def history_exog(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=94:
        raise ValueError("Expected the v1 94-column EPFL feature packet")
    return np.concatenate([x[:,14:64].reshape(-1,10,5),x[:,64:94].reshape(-1,10,3)],axis=-1)


def training_stats(x,train):
    values=history_exog(x)[train].astype(np.float64)
    mean=values.mean(axis=(0,1));scale=values.std(axis=(0,1))
    scale[scale<1e-6]=1
    mean[-1]=0;scale[-1]=1
    return dict(mean=mean.astype(np.float32),scale=scale.astype(np.float32))


def normalized_inputs(x,indices,stats,condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown EPFL information condition")
    selected=np.asarray(x[indices],dtype=np.float32)
    exog=(history_exog(selected)-stats["mean"])/stats["scale"]
    if condition=="power":exog[:]=0
    if condition=="power_state":exog[:,:,5:]=0
    if not (np.isfinite(selected).all() and np.isfinite(exog).all()):
        raise ValueError("Unexpected nonfinite EPFL input")
    n=len(selected)
    return dict(insample_y=selected[:,:10,None],
                insample_mask=np.ones((n,10,1),dtype=np.float32),
                hist_exog=exog,
                stat_exog=selected[:,10:14],
                futr_exog=np.empty((n,16,0),dtype=np.float32))

