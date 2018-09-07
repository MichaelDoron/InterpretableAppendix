from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging
torch.backends.cudnn.enabled = False

def test(args, shared_model, env_conf):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    sparsity_log = {}
    
    sparsity_log_name = ''
    if (args.conv_sparsity) and not (args.conv_app):
        sparsity_log_name = '_sparse_conv_log'
    elif (args.lstm_sparsity) and not (args.lstm_app):
        sparsity_log_name = '_sparse_lstm_log'
    elif (args.conv_sparsity) and (args.conv_app):
        sparsity_log_name = '_sparse_conv_app_log'
    elif (args.lstm_sparsity) and (args.lstm_app):
        sparsity_log_name = '_sparse_lstm_app_log'
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    setup_logger('{}{}'.format(args.env, sparsity_log_name), r'{0}{1}{2}'.format(
        args.log_dir, args.env, sparsity_log_name))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    sparsity_log['{}{}'.format(args.env, sparsity_log_name)] = logging.getLogger('{}{}'.format(
        args.env, sparsity_log_name))    
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)

    if (args.load_appendix):
        if args.conv_app:
            player.model.add_app(env.observation_space.shape[0])
        elif args.lstm_app:
            player.model.add_lstm_app(env.observation_space.shape[0])
    
    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    while True:                
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        reward_sum += player.reward

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            
            to_regularise = []
            l1_loss = 0
            
            for name, param in player.model.named_parameters():
                if (args.conv_sparsity):
                    if 'conv' in name:
                        to_regularise.append(param.view(-1))
                if (args.lstm_sparsity):
                    if ('lstm' in name) and ('app2' not in name):
                        to_regularise.append(param.view(-1))                    
                if (args.conv_app):
                    if 'app3' in name:
                        appendix_final_weights = param.view(-1)
                if (args.lstm_app):
                    if 'app4' in name:
                        appendix_final_weights = param.view(-1)
            l1_loss = torch.sum(torch.abs(torch.cat(to_regularise)))
            if (args.load_appendix):
                sparsity_log['{}{}'.format(args.env, sparsity_log_name)].info(
                    "{0:.4f}, {1}, {2}".
                    format(reward_mean, l1_loss, appendix_final_weights.abs().mean()))
                log['{}_log'.format(args.env)].info(
                    "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}, l1_loss {4}, appendix_final_weights = {5}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        reward_sum, player.eps_len, reward_mean, l1_loss, appendix_final_weights.abs().mean()))
            else:
                sparsity_log['{}{}'.format(args.env, sparsity_log_name)].info(
                    "{0:.4f}, {1}".
                    format(reward_mean, l1_loss))
                log['{}_log'.format(args.env)].info(
                    "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}, l1_loss {4}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time)),
                        reward_sum, player.eps_len, reward_mean, l1_loss))
    
            if args.save_max and (((num_tests % 10) == 0)):
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        if args.conv_sparsity or args.lstm_sparsity:
                            if args.lstm_app:
                                torch.save(state_to_save, '{0}{1}_sparse_lstm_{2}.dat'.format(
                                    args.save_model_dir, args.env, num_tests))
                            elif args.conv_app:
                                torch.save(state_to_save, '{0}{1}_sparse_conv_{2}.dat'.format(
                                    args.save_model_dir, args.env, num_tests))
                            elif args.conv_sparsity:
                                torch.save(state_to_save, '{0}{1}_sparse_conv_no_app_{2}.dat'.format(
                                    args.save_model_dir, args.env, num_tests))
                            elif args.lstm_sparsity:
                                torch.save(state_to_save, '{0}{1}_sparse_lstm_no_app_{2}.dat'.format(
                                    args.save_model_dir, args.env, num_tests))
                        else:
                            torch.save(state_to_save, '{0}{1}.dat'.format(
                                args.save_model_dir, args.env))
                            
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(
                        args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
