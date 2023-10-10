# _*_ coding: utf-8 _*_
import os


def write_log(logs_path, log_str, prefix="valid", should_print=True, mode="a"):
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)
    """ prefix: valid, train, test """
    if prefix in ["valid", "test"]:
        with open(os.path.join(logs_path, "valid_console.txt"), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if prefix in ["valid", "test", "train"]:
        with open(os.path.join(logs_path, "train_console.txt"), mode) as fout:
            if prefix in ["valid", "test"]:
                fout.write("=" * 10)
            fout.write(log_str + "\n")
            fout.flush()
    else:
        with open(os.path.join(logs_path, "%s.txt" % prefix), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if should_print:
        print(log_str)

