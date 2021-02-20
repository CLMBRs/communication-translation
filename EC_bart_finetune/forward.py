from util import idx_to_emb, logit_to_acc

import torch
from torch.autograd import Variable


# Xuhui: Whether we need this extra function here now?
def forward_joint(batch, model, loss_dict_, args, loss_fn, num_dist):

    # TODO: Why are en_batch and l2_batch set to be the same thing? I suppose we
    # don't really have one of each since the batch is just the images?
    targets = batch['target']
    en_batch = (
        batch['speaker_image'], batch['listener_images'],
        batch['speaker_caps_in'], batch['speaker_cap_lens']
    )
    l2_batch = en_batch
    output_en, output_l2, comm_actions, end_loss_, len_info = model(
        en_batch, args.sample_how
    )

    # Include the ground-truth img
    num_dist += 1

    final_loss = 0

    # TODO: We either need to figure out what `lenlen` was being used for, or
    # just delete these blocks because right now this is nonsensical and the
    # blocks never execute
    lenlen = False
    if lenlen:
        en_spk_loss = loss_fn['xent'](
            torch.index_select(
                output_en.reshape(output_en.size(0) * output_en.size(1), -1), 0,
                end_loss_[0]
            ), end_loss_[1]
        )
    else:
        en_spk_loss = torch.tensor(0).float().cuda()
    loss_dict_["average_len"].update(len_info[1].data)
    if args.loss_type == "xent":
        l2_diff_dist = torch.mean(torch.pow(output_l2[0] - output_l2[1], 2),
                                  2).view(-1, num_dist)
        l2_logits = 1 / (l2_diff_dist + 1e-10)
        l2_lsn_loss = loss_fn['xent'](l2_logits, batch['target'])
        l2_lsn_acc = logit_to_acc(l2_logits, batch['target']) * 100
        final_loss += l2_lsn_loss
    else:
        en_diff_dist = torch.mean(
            torch.pow(output_en[1][0] - output_en[1][1], 2), 2
        ).view(-1, args.num_dist)
        en_logits = 1 / (en_diff_dist + 1e-10)
        en_lsn_acc = logit_to_acc(en_logits, targets) * 100

        en_diff_dist = torch.masked_select(
            en_diff_dist,
            idx_to_emb(targets.cpu().data.numpy(),
                       args.num_dist).to(args.device)
        )
        en_lsn_loss = loss_fn['mse'](
            en_diff_dist,
            Variable(
                torch.FloatTensor(en_diff_dist.size()).fill_(0).to(args.device),
                requires_grad=False
            )
        )

        l2_diff_dist = torch.mean(
            torch.pow(output_l2[1][0] - output_l2[1][1], 2), 2
        ).view(-1, args.num_dist)
        l2_logits = 1 / (l2_diff_dist + 1e-10)
        l2_lsn_acc = logit_to_acc(l2_logits, l2_batch[7]) * 100

        l2_diff_dist = torch.masked_select(
            l2_diff_dist,
            idx_to_emb(l2_batch[7].cpu().data.numpy(),
                       args.num_dist).to(args.device)
        )
        l2_lsn_loss = loss_fn['mse'](
            l2_diff_dist,
            Variable(
                torch.FloatTensor(l2_diff_dist.size()).fill_(0).to(args.device),
                requires_grad=False
            )
        )

        final_loss += en_lsn_loss
        final_loss += l2_lsn_loss
    loss_dict_["accuracy"].update(l2_lsn_acc)
    loss_dict_["loss"].update(l2_lsn_loss.data)
    return final_loss, comm_actions
