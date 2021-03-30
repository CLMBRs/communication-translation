def checkpoint_stats2string(step, avg_loss_dict, mode="train"):
    prt_msg = "step {:5d} {} ".format(step, mode)
    prt_msg += "| loss"
    prt_msg += " {:.4f}".format(avg_loss_dict["loss"])
    # prt_msg += "| prediction accuracy"
    # prt_msg += " {:.2f}%".format(avg_loss_dict["accuracy"])
    # prt_msg += "| average message length"
    # prt_msg += " {:.4f}".format(avg_loss_dict["average_len"])
    prt_msg += " |"
    return prt_msg