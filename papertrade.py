from flask import Flask, Blueprint, render_template, request, redirect, url_for, session
import uuid
import yfinance as yf

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

paper_trading = Blueprint('paper_trading', __name__, template_folder='templates')
trades = []

def get_option_price(symbol, expiration, strike, option_type):
    try:
        ticker = yf.Ticker(symbol)
        opt_chain = ticker.option_chain(expiration)
        options = opt_chain.calls if option_type == 'call' else opt_chain.puts
        row = options[options['strike'] == strike]
        if not row.empty:
            bid = float(row['bid'].iloc[0])
            ask = float(row['ask'].iloc[0])
            last = float(row['lastPrice'].iloc[0])
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2, 2)
            else:
                return round(last, 2)
        return None
    except Exception as e:
        print(f"Error fetching option price: {e}")
        return None

def get_expirations(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.options
    except Exception:
        return []

@paper_trading.route('/trade', methods=['GET', 'POST'])
def trade():
    error = None
    if 'trades' not in session:
        session['trades'] = []

    # Step 1: Load expirations for a symbol
    if request.method == 'POST' and 'load_expirations' in request.form:
        symbol = request.form['symbol'].upper()
        expirations = get_expirations(symbol)
        session['symbol'] = symbol
        session['expirations'] = expirations
        return render_template('trade.html', trades=trades, error=error, expirations=expirations, symbol=symbol)

    # Step 2: Submit a trade
    if request.method == 'POST' and 'submit_trade' in request.form:
        try:
            symbol = session.get('symbol', '').upper()
            option_type = request.form['type'].lower()
            strike = float(request.form['strike'])
            quantity = int(request.form['quantity'])
            expiration = request.form['expiration']

            entry_option_price = get_option_price(symbol, expiration, strike, option_type)
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
            # Reset symbol/expirations for next trade
            session.pop('symbol', None)
            session.pop('expirations', None)
            return redirect(url_for('paper_trading.trade'))

        except Exception as e:
            error = f"Error: {e}"

    # Update real-time profit for open trades
    for trade in trades:
        if trade['status'] == 'open':
            current_option_price = get_option_price(
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
    symbol = session.get('symbol', '')
    expirations = session.get('expirations', [])

    return render_template('trade.html', trades=trades, error=error, expirations=expirations, symbol=symbol)

@paper_trading.route('/close_trade/<trade_id>', methods=['POST'])
def close_trade(trade_id):
    for trade in trades:
        if trade['id'] == trade_id and trade['status'] == 'open':
            exit_option_price = get_option_price(
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
