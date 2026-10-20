
from pathlib import Path

import numpy as np
import joblib
from hmmlearn.hmm import GaussianHMM


# ==========================================
# 1. CAU HINH
# ==========================================

LABELS = ["mo", "dong", "len", "xuong", "dung"]

FEATURE_DIR = Path("features")
MODEL_DIR = Path("models")

N_STATES = 5
N_ITER = 100


# ==========================================
# 2. KHOI TAO HMM TRAI SANG PHAI
# ==========================================


def make_lr_hmm(n_states=5):

    # Xac suat chuyen trang thai ban dau
    start = np.zeros(n_states)
    start[0] = 1.0

    # Ma tran chuyen trang thai trai sang phai
    trans = np.zeros((n_states, n_states))

    # Them prior nho de tranh hang co tong bang 0
    # 1.1 chi ap dung cho cac chuyen trang thai hop le
    trans_prior = np.ones((n_states, n_states))

    for i in range(n_states - 1):
        trans[i, i] = 0.7
        trans[i, i + 1] = 0.3

        trans_prior[i, i] = 1.1
        trans_prior[i, i + 1] = 1.1

    # Trang thai cuoi chi tu lap
    trans[-1, -1] = 1.0
    trans_prior[-1, -1] = 1.1

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        tol=1e-3,
        min_covar=1e-3,

        # Khoi tao mean va covariance
        init_params="mc",

        # Cap nhat transition, mean va covariance
        params="mc",

        # Prior rieng cho ma tran chuyen trang thai
        transmat_prior=trans_prior,

        random_state=42
    )

    model.startprob_ = start
    model.transmat_ = trans

    return model



# ==========================================
# 3. DOC DU LIEU TRAIN VA HUAN LUYEN
# ==========================================

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = {}

    print("=== BAT DAU HUAN LUYEN HMM ===")

    for label in LABELS:

        print("\n" + "=" * 45)
        print("NHAN:", label.upper())
        print("=" * 45)

        feature_dir = FEATURE_DIR / "train" / label

        feature_files = sorted(feature_dir.glob("*.npy"))

        if len(feature_files) != 6:
            raise ValueError(
                f"Nhan {label} can 6 file train, "
                f"nhung tim thay {len(feature_files)} file."
            )

        features = []
        lengths = []

        # Doc tung file va luu so frame.
        for path in feature_files:

            F = np.load(path, allow_pickle=False)

            if F.ndim != 2 or F.shape[1] != 39:
                raise ValueError(
                    f"Feature khong hop le: {path}, "
                    f"shape = {F.shape}"
                )

            if not np.isfinite(F).all():
                raise ValueError(
                    f"Phat hien NaN/inf trong {path}"
                )

            features.append(F)
            lengths.append(len(F))

            print(f"{path.name}: {F.shape}")

        # Ghep 6 file thanh mot ma tran lon.
        X = np.vstack(features)

        print("Tong shape:", X.shape)
        print("Lengths:", lengths)

        
        # Tao va huan luyen HMM
        model = make_lr_hmm(N_STATES)

        model.fit(X, lengths=lengths)

        # Kiem tra ket qua huan luyen
        transmat = model.transmat_

        print("So vong lap:", model.monitor_.iter)
        print("Da hoi tu:", model.monitor_.converged)

        print("Ma tran chuyen trang thai:")
        print(np.round(transmat, 3))

        # Kiem tra NaN va vo cuc
        if not np.isfinite(transmat).all():
            raise ValueError(
                f"Ma tran chuyen trang thai khong hop le: {label}"
            )

        # Tong moi hang phai bang 1
        row_sums = transmat.sum(axis=1)

        print("Tong tung hang:", np.round(row_sums, 5))

        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(
                f"Tong hang transmat khong bang 1: {label}"
            )

        # Kiem tra topology trai sang phai
        n = model.n_components
        allowed = np.zeros((n, n), dtype=bool)

        for i in range(n - 1):
            allowed[i, i] = True
            allowed[i, i + 1] = True

        allowed[n - 1, n - 1] = True

        if np.any(transmat[~allowed] > 1e-8):
            raise ValueError(
                f"Phat hien chuyen trang thai bi cam: {label}"
            )

        print("Topology trai sang phai: hop le.")

        # Chi luu model sau khi kiem tra hop le
        models[label] = model


    # ======================================
    # 4. LUU CAC MO HINH
    # ======================================

    output_path = MODEL_DIR / "hmm_models.joblib"

    joblib.dump(models, output_path)

    print("\n" + "=" * 45)
    print("DA HUAN LUYEN XONG 5 MO HINH HMM")
    print("Da luu tai:", output_path)
    print("=" * 45)

    print("Cac nhan:", list(models.keys()))


if __name__ == "__main__":
    main()
