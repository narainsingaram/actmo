from flask import Flask, Blueprint, render_template, request, redirect, url_for, session
import uuid
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure random key

POLYGON_API_KEY = 'xtc7xIvmlsu6TSk8tI44j5bqEhWAVA2l'

paper_trading = Blueprint('paper_trading', __name__, template_folder='templates')
trades = []

def ensure_str(val):
    if isinstance(val, list):
        return val[0]
    return val

def get_expirations(symbol):
    url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&order=asc&limit=1000&apiKey={POLYGON_API_KEY}"
    try:
        r = requests.get(url)
        data = r.json()
        if 'results' not in data:
            return []
        expirations = sorted({c['expiration_date'] for c in data['results']})
        return expirations
    except Exception as e:
        print(f"Error fetching expirations: {e}")
        return []

def get_strikes(symbol, expiration, option_type):
    url = f"<https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&expiration_date={expiration}&contract_type={option_type>[0].upper()}&order=asc&limit=1000&apiKey={POLYGON_API_KEY}"
    try:
        r = requests.get(url)
        data = r.json()
        if 'results' not in data:
            return []
        strikes = sorted({float(c['strike_price']) for c in data['results']})
        return strikes
    except Exception as e:
        print(f"Error fetching strikes: {e}")
        return []

def build_occ_option_symbol(symbol, expiration, strike, option_type):
    yy = expiration[2:4]
    mm = expiration[5:7]
    dd = expiration[8:10]
    cp = 'C' if option_type.lower() == 'call' else 'P'
    strike_int = int(round(float(strike) * 1000))
    strike_str = f"{strike_int:08d}"
    return f"{symbol.upper()}{yy}{mm}{dd}{cp}{strike_str}"

def get_option_market_price(symbol, expiration, strike, option_type):
    try:
        option_symbol = build_occ_option_symbol(symbol, expiration, strike, option_type)
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}/{option_symbol}?apiKey={POLYGON_API_KEY}"
        r = requests.get(url)
        data = r.json()
        if 'results' not in data or not data['results']:
            return None
        snap = data['results']
        bid = snap['last_quote']['bid_price']
        ask = snap['last_quote']['ask_price']
        last = snap['last_quote']['last_price']
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 2)
        elif last > 0:
            return round(last, 2)
        elif bid > 0:
            return round(bid, 2)
        elif ask > 0:
            return round(ask, 2)
        else:
            return None
    except Exception as e:
        print(f"Error fetching option price: {e}")
        return None

@paper_trading.route('/trade', methods=['GET', 'POST'])
def trade():
    error = None
    if 'trades' not in session:
        session['trades'] = []

    # Handle reset/back button
    if request.method == 'POST' and 'reset' in request.form:
        session.pop('symbol', None)
        session.pop('expirations', None)
        session.pop('expiration', None)
        session.pop('option_type', None)
        session.pop('strikes', None)
        return redirect(url_for('paper_trading.trade'))

    # Step 1: Load expirations for a symbol
    if request.method == 'POST' and 'load_expirations' in request.form:
        symbol = ensure_str(request.form.get('symbol', '')).upper()
        expirations = get_expirations(symbol)
        session['symbol'] = symbol
        session['expirations'] = expirations
        session.pop('expiration', None)
        session.pop('option_type', None)
        session.pop('strikes', None)
        return render_template('trade.html', trades=trades, error=error, expirations=expirations, symbol=symbol)

    # Step 2: Load strikes for expiration and option type
    if request.method == 'POST' and 'load_strikes' in request.form:
        symbol = ensure_str(session.get('symbol', '')).upper()
        expiration = ensure_str(request.form.get('expiration', ''))
        option_type = ensure_str(request.form.get('type', '')).lower()
        strikes = get_strikes(symbol, expiration, option_type)
        session['expiration'] = expiration
        session['option_type'] = option_type
        session['strikes'] = strikes
        expirations = session.get('expirations', [])
        return render_template('trade.html', trades=trades, error=error, expirations=expirations, symbol=symbol, expiration=expiration, option_type=option_type, strikes=strikes)

    # Step 3: Submit a trade
    if request.method == 'POST' and 'submit_trade' in request.form:
        try:
            symbol = ensure_str(session.get('symbol', '')).upper()
            expiration = ensure_str(session.get('expiration', ''))
            option_type = ensure_str(session.get('option_type', ''))
            strike = float(ensure_str(request.form.get('strike', '')))
            quantity = int(request.form['quantity'])

            entry_option_price = get_option_market_price(symbol, expiration, strike, option_type)
            if entry_option_price is None:
                raise Exception("Could not fetch option price for symbol/strike/expiration.")

            trade_id = str(uuid.uuid4())
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'type': option_type,
                'strike': strike,
                'quantity': quantity,
                'expiration': expiration,
                'entry_price': entry_option_price,
                'status': 'open',
                'exit_price': None,
                'profit': None
            }

            trades.append(trade)
            session['trades'] = trades
            # Reset for next trade
            session.pop('symbol', None)
            session.pop('expirations', None)
            session.pop('expiration', None)
            session.pop('option_type', None)
            session.pop('strikes', None)
            return redirect(url_for('paper_trading.trade'))

        except Exception as e:
            error = f"Error: {e}"

    # Update real-time profit for open trades
    for trade in trades:
        if trade['status'] == 'open':
            current_option_price = get_option_market_price(
                trade['symbol'],
                trade['expiration'],
                trade['strike'],
                trade['type']
            )
            if current_option_price is not None:
                trade['current_option_price'] = current_option_price
                trade['unrealized_profit'] = round(
                    (current_option_price - trade['entry_price']) * trade['quantity'] * 100, 2
                )
            else:
                trade['current_option_price'] = None
                trade['unrealized_profit'] = None

    # For form
    symbol = ensure_str(session.get('symbol', ''))
    expirations = session.get('expirations', [])
    expiration = ensure_str(session.get('expiration', ''))
    option_type = ensure_str(session.get('option_type', ''))
    strikes = session.get('strikes', [])

    return render_template('trade.html', trades=trades, error=error, expirations=expirations, symbol=symbol, expiration=expiration, option_type=option_type, strikes=strikes)

@paper_trading.route('/close_trade/<trade_id>', methods=['POST'])
def close_trade(trade_id):
    for trade in trades:
        if trade['id'] == trade_id and trade['status'] == 'open':
            exit_option_price = get_option_market_price(
                trade['symbol'],
                trade['expiration'],
                trade['strike'],
                trade['type']
            )
            if exit_option_price is None:
                break
            trade['exit_price'] = exit_option_price
            trade['profit'] = round(
                (exit_option_price - trade['entry_price']) * trade['quantity'] * 100, 2
            )
            trade['status'] = 'closed'
            break

    session['trades'] = trades
    return redirect(url_for('paper_trading.trade'))

app.register_blueprint(paper_trading)

if __name__ == '__main__':
    app.run(debug=True)
