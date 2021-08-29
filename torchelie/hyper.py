"""
Hyper parameters related utils. For now it contains hyper parameter search
tools with random search but it may evolve later.

::

    hp_sampler = HyperparamSearch(
        lr=ExpSampler(1e-6, 0.1),
        momentum=DecaySampler(0.5, 0.999),
        wd=ExpSampler(0.1, 1e-6),
        model=ChoiceSampler(['resnet', 'vgg'])
    )

    for run_nb in range(10):
        hps = hp_sampler.sample()
        results = train(**hps)
        hp_sampler.log_results(hps, results)


After creating the json file summing up the hyper parameter search, it can be
investigated with the viewer with
:code:`python3 -m torchelie.hyper hpsearch.json`. Then locate your browser to
:code:`http://localhost:8080`.
"""
import math
import random
import json
import copy

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    from scipy.stats import norm
    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False


class UniformSampler:
    """
    Uniform sampler.

    Args:
        low (float): lower bound
        high (float): higher bound
    """

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def sample(self) -> float:
        """
        Sample from the distribution.
        """
        return random.uniform(self.low, self.high)

    def inverse(self, x: float) -> float:
        return (x - self.low) / (self.high - self.low)


class ExpSampler(UniformSampler):
    """
    Exponential sampler (Uniform sampler over a log scale). Use it to sample
    the learning rate or the weight decay.

    Args:
        low (float): lower bound
        high (float): higher bound
    """

    def __init__(self, low, high):
        low = math.log10(low)
        high = math.log10(high)
        super(ExpSampler, self).__init__(min(low, high), max(low, high))

    def sample(self):
        """
        Sample a value.
        """
        return 10**super(ExpSampler, self).sample()

    def inverse(self, x):
        return super(ExpSampler, self).inverse(math.log10(x))


class DecaySampler(ExpSampler):
    """
    Sample a decay value. Use it for a momentum or beta1 / beta2 value or any
    exponential decay value.

    Args:
        low (float): lower bound
        high (float): higher bound
    """

    def __init__(self, low, high):
        super(DecaySampler, self).__init__(1 - low, 1 - high)

    def sample(self):
        """
        Sample a value.
        """
        return 1 - super(DecaySampler, self).sample()

    def inverse(self, x):
        return super(DecaySampler, self).inverse(1 - x)


class ChoiceSampler:
    """
    Sampler over a discrete of values.

    Args:
        choices (list): list of values
    """

    def __init__(self, choices):
        self.choices = choices

    def sample(self):
        """
        Sample a value
        """
        return random.choice(self.choices)

    def inverse(self, x):
        one_hot = [0] * len(self.choices)
        i = self.choices.index(x)
        one_hot[i] = 1
        return one_hot


class HyperparamSampler:
    """
    Sample hyper parameters. It aggregates multiple samplers to produce a set
    of hyper parameters.

    Example:

    ::
        HyperparamSampler(
            lr=ExpSampler(1e-6, 0.1),
            momentum=DecaySampler(0.5, 0.999),
            wd=ExpSampler(0.1, 1e-6),
            model=ChoiceSampler(['resnet', 'vgg'])
        )

    Args:
        hyperparams (kwargs): hyper params samplers. Names are arbitrary.
    """

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams

    def sample(self):
        """
        Sample hyperparameters.

        Returns:
            a dict containing sampled values for hyper parameters.
        """
        return {k: v.sample() for k, v in self.hyperparams.items()}

    def inverse(self, samples):
        return {k: self.hyperparams[k].inverse(v) for k, v in samples.items()}

    def vectorize(self, samples):
        invsamp = self.inverse(samples)
        vec = []
        for v in invsamp.values():
            if isinstance(v, (list, tuple)):
                vec += v
            else:
                vec.append(v)
        return vec


class GaussianSelector:
    is_available = HAS_SKLEARN

    def __init__(self, x, y):
        assert self.is_available, 'Cant use GaussianSelector without scipy'
        self.regressor = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(),
                                                  normalize_y=True)
        cache = self.read_cache()
        x += cache['x']
        y_mean = sum(y) / len(y)
        y += [y_mean] * len(cache['y'])
        self.regressor.fit(x, y)
        self.best = x[max(range(len(y)), key=lambda i: y[i])]

    @staticmethod
    def read_cache():
        try:
            with open('gp_cache.json', 'r') as f:
                dat = json.load(f)
            return {'x': [d['x'] for d in dat], 'y': [d['y'] for d in dat]}
        except:
            return {'x': [], 'y': []}

    @staticmethod
    def cache(x, y, id):
        try:
            with open('gp_cache.json', 'r') as f:
                dat = json.load(f)
        except:
            dat = []
        dat.append({'x': x, 'y': y, 'id': id})
        with open('gp_cache.json', 'w') as f:
            json.dump(dat, f)

    @staticmethod
    def clear_cache(id):
        try:
            with open('gp_cache.json', 'r') as f:
                dat = json.load(f)
        except:
            dat = []
        dat = [d for d in dat if d['id'] != id]
        with open('gp_cache.json', 'w') as f:
            json.dump(dat, f)

    def predict(self, x):
        mu, sigma = self.regressor.predict(x, return_std=True)

        xi = 0.01
        imp = mu - self.regressor.predict([self.best]) - xi
        Z = imp / (sigma + 1e-8)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei, mu


class SampledParams:

    def __init__(self, params):
        self.params = params
        import time
        self.tag = int(time.time() * 1000)
        self.on_destroy = (lambda: None)


class HyperparamSearch:
    """
    Perform hyper parameter search. Right now it just uses a random search.
    Params and results are logged to hpsearch.csv in the current directory. It
    would be cool to implement something like a Gaussian Process search or a RL
    algorithm.

    First, call sample() to get a set of hyper parameters. The, evaluate them
    on your task and get a dict of results. call log_results() with the hyper
    params and the results dict, and start again. Stop whenever you want.

    Args:
        hyperparameters (kwargs): named samplers (like for HyperparamSampler).
    """

    def __init__(self, **hyperparams):
        self.sampler = HyperparamSampler(**hyperparams)
        self.fname = 'hpsearch.json'

    def sample(self, algorithm='random', target=None):
        """
        Sample a set of hyper parameters.
        """
        known = self.read_hpsearch()
        if algorithm == 'random':
            return SampledParams(self.sampler.sample())

        if algorithm == 'gp':
            if len(known) == 0:
                return self.sample()

            y = [k.pop('result_' + target) for k in known]

            x = [self.sampler.vectorize(k) for k in known]

            tries = [self.sampler.sample() for _ in range(100)]
            inv_tries = [self.sampler.vectorize(t) for t in tries]

            gp = GaussianSelector(x, y)
            gain, expected = gp.predict(inv_tries)
            proposed_id = max(range(len(tries)), key=lambda i: gain[i])
            proposed = SampledParams(tries[proposed_id])
            gp.cache(inv_tries[proposed_id], expected[proposed_id],
                     proposed.tag)
            proposed.on_destroy = (lambda: gp.clear_cache(proposed.tag))
            return proposed

    def read_hpsearch(self):
        try:
            with open(self.fname, 'r') as f:
                return json.load(f)
        except:
            return []

    def _str(self, x):
        try:
            return float(x)
        except:
            pass

        try:
            return x.item()
        except:
            pass

        return x

    def log_result(self, hps, result):
        """
        Logs hyper parameters and results.
        """
        hps.on_destroy()
        res = self.read_hpsearch()
        full = copy.deepcopy(hps.params)
        full.update({'result_' + k: self._str(v) for k, v in result.items()})
        res.append(full)
        with open(self.fname, 'w') as f:
            json.dump(res, f)


if __name__ == '__main__':
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', default='hpsearch.json')
    opts = parser.parse_args()

    def make_html():
        with open(opts.file) as f:
            dat = json.load(f)

        dimensions = []
        for k in dat[0].keys():
            v = dat[0][k]
            if isinstance(v, float):
                dimensions.append({'label': k, 'values': [dd[k] for dd in dat]})
        print(json.dumps(dimensions))
        return """
        <html>
            <head>
                <meta charset="UTF-8">
                <title></title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script
                src="https://cdn.jsdelivr.net/npm/sorttable@1.0.2/sorttable.min.js"></script>
            <style>
            table, th, td {
              border: 1px solid black;
            }
            </style>
            </head>
            <body>
                <div id="graphOpts"></div>
                <div id="graphDiv"></div>
                <div id="table"></div>
        <script>
        var DATA = """ + json.dumps(dat) + """
        for (d in DATA) {
            DATA[d]['idx'] = d
        }

        let display_opts = {};
        let hidden_rows = new Set();
        function make_graph() {
            let data = DATA.filter((_, i) => !hidden_rows.has(i));
            let DIM = Object.keys(data[0])
                .filter(k => display_opts[k] !== "Hidden")
                .map(k => {
                    if (typeof data[0][k] == 'number') {
                        if (display_opts[k] == 'Visible'
                                || display_opts[k] == undefined) {
                            return {
                                'label': k,
                                'values': data.map(d => d[k])
                            };
                        } else if (display_opts[k] == 'Log') {
                            return {
                                'label': k,
                                'values': data.map(d => Math.log10(d[k])),
                                'tickvals': data.map(d => Math.log10(d[k])),
                                'ticktext': data.map(d => d[k].toExponential(2))
                            };
                        } else if (display_opts[k] == 'Decay') {
                            return {
                                'label': k,
                                'values': data.map(d => Math.log10(1/(1-d[k]))),
                                'tickvals': data.map(d => Math.log10(1/(1-d[k]))),
                                'ticktext': data.map(d => d[k].toExponential(2))
                            };
                        }
                    } else {
                        let labels = Array.from(new Set(data.map(d => d[k])));

                        return {
                            'label': k,
                            'values': data.map(d => labels.indexOf(d[k])),
                            'tickvals': labels.map((_, i) => i),
                            'ticktext': labels
                        };
                    }
                });
            var trace = {
              type: 'parcoords',
              line: {
                color: 'blue'
              },

              dimensions: DIM
            };

            var show = [trace]

            Plotly.purge('graphDiv');
            Plotly.plot('graphDiv', show, {}, {showSendToCloud: true});
        }
        make_graph();

        function change_opts(t, val) {
            display_opts[t]=val;
            make_graph();
        }

        graphOpts.innerHTML = `
            <table width="100%">
                <tr>
                    ${get_titles().map(t => `<th>${t}</th>`).join('')}
                </tr>
                <tr>
                    ${get_titles().map(t =>
                        `<td>
                            <select onchange="change_opts('${t}', event.target.value)">
                                <option default>Visible</option>
                                <option>Hidden</option>
                                <option>Log</option>
                                <option>Decay</option>
                            </select>
                        </th>`
                    ).join('')}
                </tr>
            </table>
        `;

        function get_titles() {
            let titles = [];
            for (d of DATA) {
                for (k of Object.keys(d)) {
                    titles.push(k);
                }
            }
            titles = Array.from(new Set(titles));
            return titles;
        }

        function change_visibility(r, hid) {
            if (hid) {
                hidden_rows.add(r);
            } else {
                hidden_rows.delete(r);
            }
            make_graph();
        }
        let make_table = (DATA) => {
            let titles = get_titles();
            return `
                <table width="100%" class="sortable">
                    <tr>
                    ${titles.map(t =>
                        `<th>${t}</th>`
                    ).join('')}
                    <th>Hide</th>
                    </tr>
                ${DATA.map((d, i) =>
                    `<tr>${titles.map(t =>
                        `<td>
                            ${(typeof d[t] == 'number')
                                ? d[t].toFixed(5)
                                : d[t]}
                        </td>`
                        ).join('')}
                        <td>H
                            <input type="checkbox"
                            onchange="change_visibility(${i}, event.target.checked)"/>
                        </td>
                    </tr>`).join('')}
            </table>`;
        };
        table.innerHTML = make_table(DATA);
        </script>
        </body>
        </html>
        """

    class Server(BaseHTTPRequestHandler):

        def do_GET(self):
            self.handle_http(200, 'text/html', make_html())

        def handle_http(self, status, content_type, content):
            self.send_response(status)
            self.send_header('Content-type', content_type)
            self.end_headers()
            self.wfile.write(bytes(content, 'utf-8'))

    httpd = HTTPServer(('0.0.0.0', 8080), Server)
    try:
        print('serving on localhost:8080')
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
