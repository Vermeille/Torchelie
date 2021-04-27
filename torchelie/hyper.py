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


class Sampler:
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
        return x


class ExpSampler(Sampler):
    """
    Exponential sampler (Uniform sampler over a log scale). Use it to sample
    the learning rate or the weight decay.

    Args:
        low (float): lower bound
        high (float): higher bound
    """
    def __init__(self, low, high):
        low = self.inverse(low)
        high = self.inverse(high)
        super(ExpSampler, self).__init__(low, high)

    def sample(self):
        """
        Sample a value.
        """
        return 10**super(ExpSampler, self).sample()

    def inverse(self, x):
        return math.log10(x)


class DecaySampler(ExpSampler):
    """
    Sample a decay value. Use it for a momentum or beta1 / beta2 value or any
    exponential decay value.

    Args:
        low (float): lower bound
        high (float): higher bound
    """
    def __init__(self, low, high):
        super(DecaySampler, self).__init__(low, high)

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
        return self.choices.index(x)


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

    def sample(self):
        """
        Sample a set of hyper parameters.
        """
        return self.sampler.sample()

    def read_hpsearch(self):
        try:
            with open('hpsearch.json', 'r') as f:
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
        res = self.read_hpsearch()
        full = copy.deepcopy(hps)
        full.update({'result_' + k: self._str(v) for k, v in result.items()})
        res.append(full)
        with open('hpsearch.json', 'w') as f:
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

        dat.sort(key=lambda x: x['result_lfw_loss'])
        dat = dat
        dimensions = []
        for k in dat[0].keys():
            v = dat[0][k]
            if isinstance(v, float):
                dimensions.append({
                    'label': k,
                    'values': [dd[k] for dd in dat]
                })
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
                                ? d[t].toFixed(3)
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
