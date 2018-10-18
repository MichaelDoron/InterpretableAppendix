from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from utils import WeightClipper
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import numpy as np
torch.backends.cudnn.enabled = False

def test_ablation(shared_model, args, env_conf):
    env_name = 'Pong-v0'
    # env_name = 'MsPacman-v4'
    # args = parser.parse_args('--env {} --workers 1 --gpu-ids -1 --amsgrad True --load True --conv-sparsity True --conv-app True --load-appendix True'.format(env_name).split())
    args = parser.parse_args('--env {} --workers 1 --gpu-ids -1 --amsgrad True --load True --lstm-sparsity True --lstm-app True --load-appendix True'.format(env_name).split())
    # args = parser.parse_args('--env {} --workers 1 --gpu-ids 3 --amsgrad True --load True --lstm-sparsity True --lstm-app True --load-appendix True'.format(env_name).split())
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    if (args.conv_app):
        shared_model.add_app(env.observation_space.shape[0])
    elif (args.lstm_app):
        shared_model.add_lstm_app(env.observation_space.shape[0])
    
    app_l1_losses = []
    app_mean_rewards = []
    no_app_l1_losses = []
    no_app_mean_rewards = []    
    import os
    for file in os.listdir(args.load_model_dir):
        if '{}_sparse_lstm_'.format(env_name) in file and 'no_app' not in file:
            print(file)
            saved_state = torch.load('{}{}'.format(args.load_model_dir, file),
            map_location=lambda storage, loc: storage)
            
            shared_model.load_state_dict(saved_state)
            env = atari_env(args.env, env_conf, args)
            env.seed(args.seed)
            player = Agent(None, env, args, None)
            
            player.eps_len += 2
            player.model = shared_model
            player.state = player.env.reset()
            player.state = torch.from_numpy(player.state).float()
            
            to_regularise = []
            for name, param in player.model.named_parameters():
                # if 'conv' in name:
                if ('lstm' in name) and ('app2' not in name):
                    to_regularise.append(param.view(-1))
            l1_loss = torch.sum(torch.abs(torch.cat(to_regularise)))            
            print('l1_loss is {}'.format(l1_loss))
            app_l1_losses.append(l1_loss)
            
            episode = 0
            reward_sum = 0
            reward_total_sum = 0
            reward_mean = 0
            while(episode < 10):
                player.model.eval()
                player.action_test()
                reward_sum += player.reward
                if player.done and not player.info:
                    print('Done episode {}'.format(episode))
                    state = player.env.reset()
                    player.state = torch.from_numpy(state).float()
                    reward_total_sum += reward_sum
                    reward_sum = 0
                    episode += 1
                elif player.info:
                    print('Done episode {}'.format(episode))
                    state = player.env.reset()
                    time.sleep(2)
                    player.state = torch.from_numpy(state).float()                
                    reward_total_sum += reward_sum
                    reward_sum = 0
                    episode += 1
            reward_mean = reward_total_sum / episode
            app_mean_rewards.append(reward_mean)
            
            shared_model.remove_app()
            player.state = player.env.reset()
            player.state = torch.from_numpy(player.state).float()
            
            to_regularise = []
            for name, param in player.model.named_parameters():
                # if 'conv' in name:
                if ('lstm' in name) and ('app2' not in name):
                    to_regularise.append(param.view(-1))
            l1_loss = torch.sum(torch.abs(torch.cat(to_regularise)))            
            print('l1_loss is {}'.format(l1_loss))
            no_app_l1_losses.append(l1_loss)
            
            episode = 0
            reward_sum = 0
            reward_total_sum = 0
            reward_mean = 0
            while(episode < 10):
                player.model.eval()
                player.action_test()
                reward_sum += player.reward
                if player.done and not player.info:
                    print('Done episode {}, reward {}'.format(episode, reward_sum))
                    state = player.env.reset()
                    player.state = torch.from_numpy(state).float()
                    reward_total_sum += reward_sum
                    reward_sum = 0
                    episode += 1
                elif player.info:
                    print('Done episode {}, reward {}'.format(episode, reward_sum))
                    state = player.env.reset()
                    time.sleep(2)
                    player.state = torch.from_numpy(state).float()                
                    reward_total_sum += reward_sum
                    reward_sum = 0
                    episode += 1
            reward_mean = reward_total_sum / episode
            no_app_mean_rewards.append(reward_mean)
                    


def eval(shared_model, args, env_conf):
    # env_name = 'Pong-v0'
    env_name = 'MsPacman-v4'
    # args = parser.parse_args('--env MsPacman-v4 --workers 1 --gpu-ids -1 --amsgrad True --load True --conv-sparsity True --conv-app True --load-appendix True'.format(env_name).split())
    args = parser.parse_args('--env {} --workers 1 --gpu-ids -1 --amsgrad True --load True --lstm-sparsity True --lstm-app True --load-appendix True'.format(env_name).split())
    # args = parser.parse_args('--env {} --workers 1 --gpu-ids 3 --amsgrad True --load True --lstm-sparsity True --lstm-app True --load-appendix True'.format(env_name).split())
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    if (args.conv_app):
        shared_model.add_app(env.observation_space.shape[0])
    elif (args.lstm_app):
        shared_model.add_lstm_app(env.observation_space.shape[0])
    if (args.conv_app):
        saved_state = torch.load(
        '{0}{1}_sparse_conv.dat'.format(args.load_model_dir, args.env),
        map_location=lambda storage, loc: storage)
    elif (args.lstm_app):
        saved_state = torch.load(
        '{0}{1}_sparse_lstm.dat'.format(args.load_model_dir, args.env),
        map_location=lambda storage, loc: storage)
    shared_model.load_state_dict(saved_state)
    
    env = atari_env(args.env, env_conf, args)
    env.seed(args.seed)
    player = Agent(None, env, args, None)
    
    player.eps_len += 2
    player.model = shared_model
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    
    if (args.conv_app):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        states = []
        feature_maps = []
        for r in range(1000):
            print(r)
            player.model.eval()
            player.action_test()
            states.append(np.copy(player.state.detach().cpu().numpy()))
            
            feature_maps.append([])
            for i in range(4):
                feature_maps[-1].append(np.copy(player.model.app1(Variable(player.state.unsqueeze(0)))[0, i, :,:].detach().cpu().numpy()))
            if player.done and not player.info:
                print('Done')
                state = player.env.reset()
                player.state = torch.from_numpy(state).float()
            elif player.info:
                print('player.info')
                state = player.env.reset()
                time.sleep(10)
                player.state = torch.from_numpy(state).float()
            plt.figure()
            plt.imshow(state[-1])
            plt.savefig('state_{}.png'.format(r))
            plt.close('all')
            
            for i in range(4):
                plt.figure()
                plt.imshow(feature_maps[-1][i])
                plt.savefig('state_{}_kernel_{}.png'.format(r, i))
                plt.close('all')
    
    if (args.lstm_app):
        states = []
        sentences = []
        sentiments = []
        for r in range(5000):
            player.model.eval()
            player.action_test()
            print("r = {}".format(r))
            if player.done and not player.info:
                print('Done')
                state = player.env.reset()
                player.state = torch.from_numpy(state).float()
            elif player.info:
                print('player.info')
                state = player.env.reset()
                time.sleep(2)
                player.state = torch.from_numpy(state).float()
            if ('hydrogen' in player.sentences[-1]): continue
            states.append(np.copy(player.state.detach().cpu().numpy()))
            sentences.append(player.sentences[-1])
            sentiments.append(player.model.app2.analyser.polarity_scores(sentences[-1]))
            if (sentiments[-1]['pos'] > 0) or (sentiments[-1]['neg'] > 0):
                print(sentences[-1].replace(' .', '.').replace('<start>','').replace('<uck>',''))
                print(sentiments[-1])
                plt.figure()
                plt.imshow(states[-1][0,:,:])
                plt.title(sentences[-1].replace(' .', '.').replace('<start>','').replace('<uck>',''))
                plt.savefig('{}_{}.png'.format(env_name, sentences[-1].replace(' .', '.').replace('<start>','').replace('<uck>','')))
                plt.close('all')
    
    import pickle                
    word2idx = pickle.load(open('word2idx.pkl','rb'))
    embeddings = []
    for s in sentences:
        e = torch.zeros(size=(1, player.model.app2.embed.embedding_dim))
        for w in s.split():
            e += player.model.app2.embed(torch.tensor(word2idx[w]))
        embeddings.append(e.detach().cpu().numpy())
    embeddings = np.array(embeddings).reshape(len(embeddings), player.model.app2.embed.embedding_dim)
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    
    chosen_indices = range(4134)
    clustered_embeddings = embeddings[chosen_indices]
    clustered_sentences = np.array(sentences)[chosen_indices]
    clustered_states = np.array(states)[chosen_indices]
    X_embedded = tsne.fit_transform(clustered_embeddings)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.savefig('LSTM_wordsss.png')
    
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=3, min_samples=10).fit(X_embedded)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
    X = X_embedded
    plt.rcParams['axes.facecolor'] = 'black'
    fig1, ax = plt.subplots(nrows=1, ncols=1)
    fig1.patch.set_facecolor('black')
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.savefig('DBscan.png', facecolor=fig1.get_facecolor())
    plt.close('all')
    
    label_sizes = []
    for label in sorted(unique_labels):
        label_sizes.append(np.where(labels == label)[0].shape[0])
    
    sorted_label = np.array(sorted(unique_labels))[np.argsort(label_sizes)][::-1]
    
    np.array(clustered_sentences)[np.where(labels == sorted_label[label])[0]][0]
    
    plt.close('all')
    for label in sorted(unique_labels):
        (unique_sentences, unique_indices) = np.unique(np.array(clustered_sentences)[np.where(labels == sorted_label[label])[0]], return_index = True)
        for state_ind in unique_indices:
            plt.figure()
            plt.imshow(np.array(clustered_states)[np.where(labels == sorted_label[label])[0]][state_ind][0,:,:])
            plt.title(np.array(clustered_sentences)[np.where(labels == sorted_label[label])[0]][state_ind])
            plt.savefig('{}_label_{}_state_{}.png'.format(env_name, label, state_ind))
            plt.close('all')
            print(np.array(clustered_sentences)[np.where(labels == sorted_label[label])[0]][state_ind])
        print('label: {}, pos: {}'.format(label, player.model.app2.analyser.polarity_scores(np.array(clustered_sentences)[np.where(labels == sorted_label[label])[0]][0])['pos']))
    
    for ind in range(len(sentences)):
        if ('hydrogen' in sentences[ind]): continue
        if (sentiments[ind]['pos'] > 0) or (sentiments[ind]['neg'] > 0):
            print(sentences[ind].replace(' .', '.').replace('<start>','').replace('<uck>',''))
            print(sentiments[ind])
            plt.figure()
            plt.imshow(states[ind][0,:,:])
            plt.title(sentences[ind].replace(' .', '.').replace('<start>','').replace('<uck>',''))
            plt.savefig('{}_{}.png'.format(env_name, sentences[ind].replace(' .', '.').replace('<start>','').replace('<uck>','')))
            plt.close('all')
            
    
def train(rank, args, shared_model, optimizer, env_conf):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)
    
    teacher = Agent(None, env, args, None)
    teacher.gpu_id = gpu_id
    teacher.model = A3Clstm(teacher.env.observation_space.shape[0],
                           teacher.env.action_space)
    
    if (args.load_appendix):
        if args.conv_app:
            player.model.add_app(env.observation_space.shape[0])
        elif args.lstm_app:
            player.model.add_lstm_app(env.observation_space.shape[0])
    
    if (args.load_appendix):
        if args.conv_app:
            teacher.model.add_app(env.observation_space.shape[0])
        elif args.lstm_app:
            teacher.model.add_lstm_app(env.observation_space.shape[0])
    
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()

    teacher.state = player.env.reset()
    teacher.state = torch.from_numpy(teacher.state).float()
    
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            teacher.state = teacher.state.cuda()
            teacher.model = teacher.model.cuda()

    player.model.train()
    player.eps_len += 2


    l1_weight = 0.0
    T = 0
    last_ten_R = []
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            teacher.model.load_state_dict(shared_model.state_dict())
    else:
        teacher.model.load_state_dict(shared_model.state_dict())

    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())

        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 512).cuda())
                    player.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 512))
                player.hx = Variable(torch.zeros(1, 512))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        if teacher.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    teacher.cx = Variable(torch.zeros(1, 512).cuda())
                    teacher.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                teacher.cx = Variable(torch.zeros(1, 512))
                teacher.hx = Variable(torch.zeros(1, 512))
        else:
            teacher.cx = Variable(teacher.cx.data)
            teacher.hx = Variable(teacher.hx.data)


        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        for step in range(args.num_steps):
            teacher.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        if teacher.done:
            state = teacher.env.reset()
            teacher.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    teacher.state = teacher.state.cuda()


        R = torch.zeros(1, 1)
        if not teacher.done:
            value, _, _ = teacher.model((Variable(teacher.state.unsqueeze(0)),
                                        (teacher.hx, teacher.cx)))
            R = value.data

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                        (player.hx, player.cx)))
            R = value.data


        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        teacher.values.append(Variable(R))

        policy_loss = 0
        value_loss = 0
        snt_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            
            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data
            
            gae = gae * args.gamma * args.tau + delta_t
            
            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]
        
        player.model.zero_grad()
        for snt in player.snts:
            snt_loss += snt
        total_loss = policy_loss + 0.5 * value_loss + snt_loss
        
        if (args.lstm_sparsity):
            if (((T % int(1000 / args.workers)) == 0) and (l1_weight <= 0.01)):
                l1_weight += 0.000005
                print('l1_weight = {}'.format(l1_weight))
        elif (args.conv_sparsity):
            if (((T % int(1000 / args.workers)) == 0) and (l1_weight <= 0.01)):
                l1_weight += 0.005
                print('l1_weight = {}'.format(l1_weight))
        T = T + 1
        
        to_regularise = []
        for name, param in player.model.named_parameters():
            if (args.conv_sparsity):
                if 'conv' in name:
                    to_regularise.append(param.view(-1))
            if (args.lstm_sparsity):
                if ('lstm' in name) and ('app2' not in name):
                    to_regularise.append(param.view(-1))
        l1_loss = l1_weight * torch.sum(torch.abs(torch.cat(to_regularise)))
        total_loss += l1_loss
        total_loss.backward()
        # print('l1_loss: {}'.format(l1_loss))
        
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        
        clipper = WeightClipper()
        player.model.apply(clipper)
        
        player.clear_actions()
