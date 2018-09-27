import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
from tools.parse import thread_parse
from pickle import load
from random import randint
from itertools import permutations
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import cm

plt.style.use("dark_background")
rcParams['figure.figsize'] = (13, 8)
rcParams['font.family'] = 'monospace'


class GroupChat:

    def __init__(self, master_list):

        self.master = master_list
        self.size = len(master_list)

        self.users = sorted(list(set([m['sndr'] for m in master_list])))
        self.users_initials = [''.join(map(lambda x: x[0],
                                           u.split(' '))) for u in self.users]

        self.user_bins = self.user_sort()

        self.max_len = max(len(p) for p in self.users) + 1

        self.times = np.array([m['time'] for m in self.master],
                              dtype='datetime64[m]')

        self.sorted_master = self.message_sort(self.master)

        self.totals = [float(len(self.user_bins[i]['msgs']))
                       for i in range(len(self.users))]

        self.convos = self.cluster_find()

    def user_sort(self):
        usr_bins = [
            {'Name': n, 'msgs': [], 'texts': [],
             'dates': [], 'bins': []} for n in self.users
        ]

        for m in self.master:
            for u in usr_bins:
                if m['sndr'] == u['Name']:
                    u['msgs'].append(m)
                    u['texts'].append(m['text'])
                    u['dates'].append(m['time'])

        return usr_bins

    def cluster_find(self, threshold=30.0):
        cluster_ix, j = [0], 0

        for i in range(1, len(self.sorted_master)):

            td = (self.sorted_master[i]['time'] -
                  self.sorted_master[i - 1]['time'])

            if td.total_seconds() / 60.0 > threshold:
                j += 1
            cluster_ix.append(j)

        clusters = [{'msgs': []} for i in range(max(cluster_ix) + 1)]
        for i, m in enumerate(self.sorted_master):
            clusters[cluster_ix[i]]['msgs'].append(m)

        return clusters

    def conversation_matrix(self):
        convo_matrix = np.zeros(shape=(len(self.users),
                                       len(self.users)))

        for convo in self.convos:

            members = list(set([c['sndr'] for c in convo['msgs']]))
            ids = [self.users.index(m) for m in members]

            if len(ids) > 1:
                perms = np.array(list(permutations(ids, 2)))

                convo_matrix[perms[:, 0], perms[:, 1]] += 1.0

        for i in range(len(self.users)):
            convo_matrix[i, :] /= (self.totals[i])
            convo_matrix[i, :] /= (convo_matrix[i, :].sum()) / 100.0

        return convo_matrix

    @staticmethod
    def message_sort(msgs, reverse=False):
        mtimes = np.array([m['time'] for m in msgs],
                          dtype='datetime64[m]')

        return [y for (x, y) in sorted(zip(mtimes, msgs),
                                       reverse=reverse)]

    def random(self, n=5):
        ix = [randint(0, len(self.master)) for _ in range(n)]
        for i in ix:
            print self.message_string(self.master[i])

    def message_string(self, msg):
        return (
            '{date} {name:{width}} {text}'.format(
                name=msg['sndr'], width=self.max_len,
                date=msg['time'].strftime('%d/%m/%y %H:%M'),
                text=msg['text'].encode('utf-8'))
        )

    def message_rank(self):
        counts = [len(u['msgs']) for u in self.user_bins]
        ranked_users = sorted(self.users, key=dict(zip(self.users, counts)).get, reverse=True)
        ranked_counts = sorted(counts, reverse=True)
        percentiles = map(lambda x: 100.0 * x / len(self.master), ranked_counts)
        for x, y, p in zip(ranked_users, ranked_counts, percentiles):
            print "name: " + unicode(x) + "  count: " + unicode(y) + "  percentage: " + unicode(p)
        o = 17
        explode = np.zeros(o+1)
        explode[1] = 0.1
        others = 0
        for l in range(len(ranked_counts) - o):
            others += ranked_counts[o + l]
        ranked_users[o] = str(len(ranked_counts) - o) + " others"
        ranked_counts[o] = others
        fig, ax = plt.subplots()
        ax.pie(ranked_counts[:o+1], explode=explode, labels=ranked_users[:o+1], autopct="%1.1f%%", shadow=True, startangle=90, colors=["#ff9999", "#66b3ff", "#00b100", "#ffcc99"])
        ax.axis("equal")
        fig.tight_layout()
        fig = self.fig_watermark(fig, '% of comments submitted')
        fig.savefig('./plots/pie.png')


    def time_bins(self, times_list, bin_size=1):
        all_days = np.array(self.times - self.times.min(), dtype='timedelta64[D]').astype('int')
        days = (times_list - self.times.min()).astype('timedelta64[D]').astype('int')
        bins = np.histogram(days, range(1, all_days.max() + 3, bin_size))[0]
        time = np.arange(self.times.min(), self.times.max(),
                         bin_size, dtype='datetime64[D]')

        if len(time) != len(bins):
            return time, bins[:len(time)]
        else:
            return time, bins

    def time_plot(self, bin_size=1, window=30):

        print 'Plotting total group activity with time'
        timex, timey = self.time_bins(self.times, bin_size)

        avg = np.convolve(timey, np.ones((window,)) / float(window), mode='same')

        fig, ax = plt.subplots()
        ax.plot(timex.astype(datetime), timey[:len(timex)] / bin_size, color='C0', alpha=0.1)
        ax.plot(timex.astype(datetime), avg / bin_size, color='C0')
        ax.set_xlabel('Date')
        ax.set_ylabel('Avg. Messages per Day')
        ax.set_ylim([0, np.max(avg / bin_size) * 1.1])

        fig.tight_layout()
        fig = self.fig_watermark(fig, 'All-User Lifetime Activity')
        fig.set_size_inches(30, 10.8)
        fig.savefig('./plots/all-user_lifetime_activity_wide.png')

    def time_plot_user(self, names, bin_size=1, window=30):
        print 'Plotting individual activity with time'
        data = []
        for name in names:
            for user in self.user_bins:
                if user['Name'] == name:
                    data.append(self.time_bins(
                        np.array(user['dates'], dtype='datetime64[m]')
                    ))
        fig, ax = plt.subplots()
        color = iter(cm.rainbow(np.linspace(0, 1, len(names))))
        linestyles = ["-", "--", "-.", ":"]
        a = 0
        for i in range(len(names)):
            c = next(color)
            avg = np.convolve(data[i][1], np.ones((window,)) / window, mode='same')
            if len(names) == 1:
                ax.plot(data[i][0].astype(datetime), avg / bin_size, label=names[i], c="white", linestyle=linestyles[a])
            else:
                ax.plot(data[i][0].astype(datetime), avg / bin_size, label=names[i], c=c, linestyle=linestyles[a])
            a+= 1
            if a == 4:
                a = 0
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Avg. Messages per Day')
        fig.tight_layout()

        fig.set_size_inches(30, 10.8)
        if len(names) == 1:
            fig = self.fig_watermark(fig, unicode(names[0]) + ' Lifetime Activity')
            fig.savefig('./plots/' + unicode(names[0]) + '_lifetime_activity_wide.png')
        else:
            fig = self.fig_watermark(fig, '%d User Lifetime Activity' % len(names))
            fig.savefig('./plots/s_lifetime_activity_wide.png')

    def matrix_plot(self):

        print 'Plotting conversation matrix'
        convo_matrix = self.conversation_matrix()

        fig, ax = plt.subplots()
        imax = ax.imshow(convo_matrix, interpolation='none')
        ax.grid(False)
        ax.set_xticks(range(0, 14))
        ax.set_xticklabels(self.users_initials, rotation=90.0)
        ax.set_yticks(range(0, 14))
        ax.set_yticklabels(self.users_initials)
        ax.set_xlabel('Person $X$')
        ax.set_ylabel('Person $Y$')
        for i in range(len(self.users)):
            for j in range(len(self.users)):
                ax.text(i, j, '%.0f' % convo_matrix[j, i],
                        color='w', va='center', ha='center')
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="3%", pad=0.5)
        cbar = plt.colorbar(imax, cax=cax1)
        cbar.set_label("Amount of $Y$'s Conversation Which is Shared With $X$ (%)",
                       labelpad=10.0)
        fig.tight_layout()
        fig = self.fig_watermark(fig, 'Conversation Matrix')
        fig.savefig('./plots/Conversation_Matrix_All_User.png')

    def word_print(self, words):
        for msg in self.master:
            if any(x in msg['text'] for x in words):
                print self.message_string(msg)

    def word_find(self, words):
        times = []
        for msg in self.master:
            if any(x in msg['text'] for x in words):
                times.append(msg['time'])
        return np.array(times, dtype='datetime64[m]')

    def word_plot(self, words, bin_size=30):
        fig, ax = plt.subplots()
        full_string = ','.join([','.join(w) for w in words]).replace(' ', '')

        # Loop over groups of words
        for w in words:
            tl = self.word_find(w)
            tb = self.time_bins(tl, bin_size=bin_size)
            t0 = self.time_bins(self.times, bin_size=bin_size)
            ax.plot(tb[0].astype(datetime), 100.0 * tb[1].astype(float) / t0[1].astype(float),
                    label=','.join(w).replace(' ', ''))

        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Occurrence Rate per %d Days (%%)' % bin_size)

        fig.tight_layout()
        fig = self.fig_watermark(fig, 'Usage for %s' % full_string)
        fig.savefig('./plots/Word_Usage_%s.png' % full_string)

    def daily_plot(self, names=None, window=60):
        print 'Plotting group daily activity'

        fig, ax = plt.subplots()
        x = np.linspace(0.0, 24.0, 1440)
        ax.set_ylabel('Messages per hour')
        ax.set_xticks(np.linspace(0.0, 24.0, 25))
        ax.set_xlabel('Hour of Day')
        if names:
            data = []
            for name in names:
                for user in self.user_bins:
                    if user['Name'] == name:
                        data.append(np.array(user['dates'],
                                             dtype='datetime64[m]'))
            color = iter(cm.rainbow(np.linspace(0, 1, len(names))))
            linestyles = ["-", "--", "-.", ":"]
            a = 0
            for i, name in enumerate(names):
                c = next(color)
                d = self.daily_bins(data[i])
                if len(names) == 1:
                    ax.plot(x, moving_average(d, window), label=name, c="white", linestyle=linestyles[a])
                else:
                    ax.plot(x, moving_average(d, window), label=name, c=c, linestyle=linestyles[a])
                ax.legend()
                a += 1
                if a == 4:
                    a = 0
            title = '%d-User Daily Activity' % len(names)
            if len(names) == 1:
                fig = self.fig_watermark(fig, unicode(names[0]) + ' Daily Activity')
                fig.set_size_inches(30, 10.8)
                fig.savefig('./plots/' + unicode(names[0]) + '_daily_activity.png')
            else:
                fig = self.fig_watermark(fig, title)
                fig.set_size_inches(30, 10.8)
                fig.savefig('./plots/%s.png' % title)
        else:
            ax.plot(x, self.daily_bins(self.times), color='k', alpha=0.1)
            ax.plot(x, moving_average(self.daily_bins(self.times), window))
            title = 'All-User Daily Activity'

            ax.set_ylabel('Messages per hour')
            ax.set_xticks(np.linspace(0.0, 24.0, 25))
            ax.set_xlabel('Hour of Day')

            fig.tight_layout()
            fig = self.fig_watermark(fig, title)
            fig.savefig('./plots/%s.png' % title)
            # fig.set_size_inches(30, 10.8)
            # fig.savefig('./plots/%s_.png' % title)


    def weekly_plot(self, window=60):
        print 'Plotting weekly activity'

        d = self.daily_bins(self.times, weekday=True)
        avg = []
        for i in range(7):
            avg.append(moving_average(d[i, :], 60))
        avg = np.array(avg)

        fig, ax = plt.subplots()
        imax = ax.imshow(avg, extent=[0, 24, 0, 0.6 * 24], origin='lower')

        ax.set_yticks(np.linspace(1.0, 0.6 * 24 - 1.0, 7))
        ax.set_xticks(np.linspace(0.0, 24.0, 25))

        ax.set_ylabel('Day of Week')
        ax.set_xlabel('Hour of Day')
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        ax.grid(False)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="3%", pad=0.5)
        cbar = plt.colorbar(imax, cax=cax1)

        cbar.set_label("Messages per Hour",
                       labelpad=10.0)
        fig.tight_layout()
        fig = self.fig_watermark(fig, 'Weekly Activity')
        fig.savefig('./plots/Group_Daily_Weekly_Activity.png')

    def message_length_plot(self):
        fig, ax = plt.subplots()

        length_dist = np.array([len(msg['text'].split(' ')) for msg in self.master])
        dist = ax.hist(length_dist, bins=np.linspace(1, 50, 50))
        ax.text(50, dist[0].max() * 0.9, 'Mean Length = %.2f Words' % length_dist.mean(),
                size=20, ha='right')
        ax.set_xlabel('Message Length (Words)')
        ax.set_ylabel('Frequency')
        fig.tight_layout()
        fig = groupchat.fig_watermark(fig, 'Message Length')
        fig.savefig('./plots/Message_Length')

    def word_length_plot(self):
        print 'Plotting Word Length Distributions'

        char_dist = []

        for i in range(len(self.users)):
            c = [[len(w) for w in msg.split(' ')] for msg in self.user_bins[i]['texts']]
            char_dist.append(np.array([x for y in c for x in y]))
        means = np.array([c.mean() for c in char_dist])
        percs = np.array([c.std() for c in char_dist])

        c = [[len(w) for w in msg['text'].split(' ')] for msg in self.master]
        c = np.array([x for y in c for x in y])

        fig, ax = plt.subplots()
        ax.axvline(c.mean(), linestyle='--', color='white', alpha=0.5)
        ax.errorbar(means, range(1, len(groupchat.users) + 1), xerr=percs, marker='o', linestyle='none',
                    capsize=5.0)
        ax.set_xlim([-1, 10])
        ax.set_yticks(range(1, len(groupchat.users) + 1))
        ax.set_yticklabels(groupchat.users)
        ax.set_xlim([0.0, 15])
        ax.set_xlabel('Word Length (chars)')
        fig.tight_layout()
        fig = groupchat.fig_watermark(fig, 'User Word Length Distribution')
        fig.savefig('./plots/All_User_Word_Length_Distribution.png')


    @staticmethod
    def daily_bins(times, weekday=False):
        if weekday:
            bins = np.zeros((7, 1440))
            for t in times:
                ix = (t.astype(datetime).hour * 60) + t.astype(datetime).minute
                bins[t.astype(datetime).weekday(), ix] += 1
        else:
            bins = np.zeros(1440)
            for t in times:
                ix = (t.astype(datetime).hour * 60) + t.astype(datetime).minute
                bins[ix] += 1
        return bins*60 / (1440.0)

    @staticmethod
    def fig_watermark(fig, title):
        fig.subplots_adjust(top=0.9)
        x = fig.axes[0].get_position().x0
        w = fig.axes[0].get_position().width
        y = 0.92
        fig.text(x, y, title, family='serif', size=30)
        fig.text(x + w, y, "P4 Facebook chat analysis",
                 family='serif', size=10, ha='right', alpha=0.4)
        return fig


def messages_load():
    if not os.path.exists('./input/group_chat.pkl') \
            or '-force' in sys.argv:
        print 'No pkl file found, parsing HTML...'
        thread_parse("message.html")

    else:
        print 'File found, loading messages...'

    with open('./input/group_chat.pkl', 'r') as f:
        return GroupChat(load(f))


def moving_average(data, window):

    dd = np.concatenate((data, data))

    return np.convolve(np.ones(window) / float(window), dd)[window: len(data) + window]


if __name__ == '__main__':

    groupchat = messages_load()

    if True:
        print 'Performing all analysis...'
        groupchat.time_plot()
        groupchat.time_plot_user(groupchat.users)
        # groupchat.matrix_plot()
        groupchat.daily_plot()
        groupchat.daily_plot(names=groupchat.users)
        groupchat.weekly_plot()
        groupchat.message_length_plot()
        groupchat.word_length_plot()
        groupchat.message_rank()
        plt.rcParams.update({'font.size': 24})
        for user in groupchat.users:
            groupchat.time_plot_user([user])
        for user in groupchat.users:
            groupchat.daily_plot(names=[user])
