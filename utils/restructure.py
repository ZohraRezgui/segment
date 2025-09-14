import argparse
import glob
import os
import shutil


def restructure_zip_folders(root, outdir):
    os.makedirs(os.path.join(outdir, "train1"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "train2"), exist_ok=True)
    train_1 = [
        "02",
        "04",
        "05",
        "06",
        "08",
        "09",
        "10",
        "11",
        "12",
        "15",
        "16",
        "17",
        "19",
    ]
    train_folders = glob.glob(os.path.join(root, "train") + "/*.zip")
    for vid_zip in train_folders:
        base = os.path.basename(vid_zip)
        vid_n = base.split("_")[1][:2]
        # print('video_', vid_n)
        if vid_n in train_1:
            shutil.copy2(vid_zip, os.path.join(outdir, "train1", base))
        else:
            shutil.copy2(vid_zip, os.path.join(outdir, "train2", base))


def verify_file_counts(out):
    train1_count = len(glob.glob(os.path.join(out, "train1", "*.zip")))
    train2_count = len(glob.glob(os.path.join(out, "train2", "*.zip")))
    assert train1_count == 16, f"Expected 16 files in train1, got {train1_count}"
    assert train2_count == 28, f"Expected 28 files in train2, got {train2_count}"
    print(f"Verified: train1={train1_count}, train2={train2_count}")


def main(root, outdir):
    restructure_zip_folders(root, outdir)
    verify_file_counts(outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="restructure the data root folder")
    parser.add_argument(
        "--root", "-r", help="path to the root folder where data is downloaded"
    )
    parser.add_argument(
        "--outdir", "-o", help="path to the output folder where data will be saved"
    )
    args = parser.parse_args()
    main(args.root, args.outdir)
