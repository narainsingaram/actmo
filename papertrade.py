from flask import Flask, Blueprint, render_template, request, redirect, url_for, session
import uuid
from datetime import datetime, timedelta, date
import pytz
import math
import random
import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import time

app = Flask(__name__)
app.secret_key = 'your_secure_random_key_12345'  # Replace with a strong, unique key
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'memory'

# Market parameters
RISK_FREE_RATE = 0.0525  # Current 10-year Treasury rate (updated)
MARKET_HOURS = {
    'open': 9.5,   # 9:30 AM EST
    'close': 16.0  # 4:00 PM EST
}

# Enhanced cache for market data
MARKET_DATA_CACHE = {}  # {symbol: {'data': dict, 'timestamp': datetime, 'expiry': datetime}}
OPTION_CHAIN_CACHE = {}  # {symbol: {'chains': dict, 'timestamp': datetime}}
TREASURY_RATE_CACHE = {'rate': RISK_FREE_RATE, 'timestamp': None}

paper_trading = Blueprint('paper_trading', __name__, template_folder='templates')

def is_market_open():
    """Check if market is currently open (simplified - doesn't account for holidays)"""
    now = datetime.now(pytz.timezone('US/Eastern'))
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    current_hour = now.hour + now.minute/60.0
    
    # Market closed on weekends
    if weekday >= 5:  # Saturday or Sunday
        return False
    
    # Check if within trading hours
    return MARKET_HOURS['open'] <= current_hour <= MARKET_HOURS['close']

def get_risk_free_rate():
    """Get current risk-free rate from Treasury data"""
    global TREASURY_RATE_CACHE
    now = datetime.now()
    
    # Update daily
    if (TREASURY_RATE_CACHE['timestamp'] is None or 
        (now - TREASURY_RATE_CACHE['timestamp']).days >= 1):
        try:
            # Try to get 10-year Treasury rate
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="5d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1] / 100  # Convert percentage to decimal
                TREASURY_RATE_CACHE = {'rate': rate, 'timestamp': now}
                print(f"Updated risk-free rate: {rate:.4f}")
            else:
                print("Using fallback risk-free rate")
        except Exception as e:
            print(f"Error fetching Treasury rate: {e}")
    
    return TREASURY_RATE_CACHE['rate']

def enhanced_black_scholes(S, K, T, r, sigma, option_type, q=0):
    """
    Enhanced Black-Scholes with dividend yield and better numerical stability
    """
    try:
        if T <= 0:
            # Handle expiration
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        if sigma <= 0:
            return 0.01
        
        # Add small buffer to avoid numerical issues
        T = max(T, 1/365)  # At least 1 day
        sigma = max(sigma, 0.01)  # At least 1% volatility
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:  # put
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        
        return max(round(float(price), 2), 0.01)
    
    except Exception as e:
        print(f"Black-Scholes calculation error: {e}")
        return 0.01

def get_enhanced_market_data(symbol, force_refresh=False):
    """Get comprehensive market data with enhanced caching"""
    global MARKET_DATA_CACHE
    now = datetime.now()
    cache_key = symbol.upper()
    
    # Check cache (5-minute expiry during market hours, 1-hour when closed)
    cache_duration = 300 if is_market_open() else 3600  # seconds
    
    if (not force_refresh and cache_key in MARKET_DATA_CACHE and 
        (now - MARKET_DATA_CACHE[cache_key]['timestamp']).seconds < cache_duration):
        return MARKET_DATA_CACHE[cache_key]['data']
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current price and basic info
        info = ticker.info
        current_price = info.get('regularMarketPrice') or info.get('currentPrice')
        
        if current_price is None:
            print(f"No price data available for {symbol}")
            return None
        
        # Get historical data for volatility calculation
        hist = ticker.history(period="60d", interval="1d")
        if len(hist) < 20:
            print(f"Insufficient historical data for {symbol}")
            return None
        
        # Calculate realized volatility (multiple methods for accuracy)
        returns = hist['Close'].pct_change().dropna()
        
        # 30-day realized volatility
        volatility_30d = returns.tail(30).std() * np.sqrt(252)
        
        # 60-day realized volatility
        volatility_60d = returns.std() * np.sqrt(252)
        
        # GARCH-like weighted volatility (more recent data weighted higher)
        weights = np.exp(np.linspace(-1, 0, len(returns)))
        weights = weights / weights.sum()
        weighted_returns = returns * weights
        volatility_weighted = np.sqrt(np.sum(weighted_returns**2) * 252)
        
        # Average the volatility estimates
        volatility = np.mean([volatility_30d, volatility_60d, volatility_weighted])
        volatility = np.clip(volatility, 0.05, 3.0)  # Reasonable bounds
        
        # Get dividend yield if available
        dividend_yield = info.get('dividendYield', 0) or 0
        
        # Get bid-ask spread approximation from volume
        avg_volume = hist['Volume'].tail(10).mean()
        bid_ask_spread = max(0.01, min(0.5, 1000000 / avg_volume))  # Rough approximation
        
        market_data = {
            'price': round(current_price, 2),
            'volatility': round(volatility, 4),
            'dividend_yield': dividend_yield,
            'bid_ask_spread': round(bid_ask_spread, 2),
            'volume': hist['Volume'].iloc[-1] if not hist.empty else 0,
            'market_cap': info.get('marketCap', 0),
            'beta': info.get('beta', 1.0),
            'last_update': now
        }
        
        # Add small random market movement during market hours
        if is_market_open():
            # Simulate intraday movement based on volatility
            daily_vol = volatility / np.sqrt(252)
            random_move = np.random.normal(0, daily_vol * 0.1)  # 10% of daily vol
            market_data['price'] = round(current_price * (1 + random_move), 2)
        
        MARKET_DATA_CACHE[cache_key] = {
            'data': market_data,
            'timestamp': now
        }
        
        print(f"Updated market data for {symbol}: Price=${market_data['price']}, Vol={volatility:.2%}")
        return market_data
        
    except Exception as e:
        print(f"Error fetching market data for {symbol}: {e}")
        return None

def get_real_option_chains(symbol):
    """Get real option expiration dates and strikes from yfinance"""
    global OPTION_CHAIN_CACHE
    now = datetime.now()
    cache_key = symbol.upper()
    
    # Cache for 1 hour
    if (cache_key in OPTION_CHAIN_CACHE and 
        (now - OPTION_CHAIN_CACHE[cache_key]['timestamp']).seconds < 3600):
        return OPTION_CHAIN_CACHE[cache_key]['chains']
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get available expiration dates
        expirations = ticker.options
        if not expirations:
            print(f"No options data available for {symbol}")
            return None
        
        # Get current stock price for strike filtering
        market_data = get_enhanced_market_data(symbol)
        if not market_data:
            return None
        
        current_price = market_data['price']
        
        option_chains = {}
        
        # Limit to first 12 expirations to avoid too much data
        for exp_date in expirations[:12]:
            try:
                option_chain = ticker.option_chain(exp_date)
                
                # Get strikes around current price (±50% range)
                price_range = current_price * 0.5
                min_strike = current_price - price_range
                max_strike = current_price + price_range
                
                # Filter calls and puts
                calls = option_chain.calls
                puts = option_chain.puts
                
                if not calls.empty and not puts.empty:
                    # Filter strikes within reasonable range
                    call_strikes = calls[(calls['strike'] >= min_strike) & 
                                       (calls['strike'] <= max_strike)]['strike'].tolist()
                    put_strikes = puts[(puts['strike'] >= min_strike) & 
                                     (puts['strike'] <= max_strike)]['strike'].tolist()
                    
                    # Combine and sort unique strikes
                    all_strikes = sorted(list(set(call_strikes + put_strikes)))
                    
                    if all_strikes:
                        option_chains[exp_date] = {
                            'strikes': all_strikes,
                            'calls': calls,
                            'puts': puts
                        }
                
            except Exception as e:
                print(f"Error processing expiration {exp_date}: {e}")
                continue
        
        if option_chains:
            OPTION_CHAIN_CACHE[cache_key] = {
                'chains': option_chains,
                'timestamp': now
            }
            print(f"Cached option chains for {symbol}: {len(option_chains)} expirations")
        
        return option_chains
        
    except Exception as e:
        print(f"Error fetching option chains for {symbol}: {e}")
        return None

def get_accurate_option_price(symbol, expiration, strike, option_type):
    """Get more accurate option pricing using real market data and improved models"""
    try:
        # Get enhanced market data
        market_data = get_enhanced_market_data(symbol)
        if not market_data:
            return None
        
        current_price = market_data['price']
        volatility = market_data['volatility']
        dividend_yield = market_data['dividend_yield']
        bid_ask_spread = market_data['bid_ask_spread']
        
        # Calculate time to expiration
        today = datetime.now().date()
        exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
        
        if exp_date <= today:
            # Handle expiration day
            if option_type.lower() == 'call':
                return max(current_price - float(strike), 0)
            else:
                return max(float(strike) - current_price, 0)
        
        days_to_exp = (exp_date - today).days
        T = days_to_exp / 365.0
        
        # Get current risk-free rate
        risk_free_rate = get_risk_free_rate()
        
        # Calculate theoretical price
        theoretical_price = enhanced_black_scholes(
            S=current_price,
            K=float(strike),
            T=T,
            r=risk_free_rate,
            sigma=volatility,
            option_type=option_type,
            q=dividend_yield
        )
        
        # Add bid-ask spread simulation
        if option_type.lower() == 'call':
            # Calls: add half spread to theoretical
            final_price = theoretical_price + (bid_ask_spread / 2)
        else:
            # Puts: subtract half spread from theoretical
            final_price = theoretical_price - (bid_ask_spread / 2)
        
        # Add small random market maker spread
        market_noise = np.random.uniform(-0.02, 0.02)  # ±2 cents
        final_price = max(final_price + market_noise, 0.01)
        
        # Round to nearest cent
        final_price = round(final_price, 2)
        
        print(f"Option price calculation for {symbol} {option_type} {strike} exp {expiration}:")
        print(f"  Underlying: ${current_price}, Vol: {volatility:.2%}, T: {T:.4f}")
        print(f"  Theoretical: ${theoretical_price}, Final: ${final_price}")
        
        return final_price
        
    except Exception as e:
        print(f"Error calculating accurate option price: {e}")
        return None

def get_available_strikes(symbol, expiration):
    """Get available strikes for a specific expiration"""
    option_chains = get_real_option_chains(symbol)
    if not option_chains or expiration not in option_chains:
        return []
    
    return option_chains[expiration]['strikes']

def update_trade_values_enhanced():
    """Enhanced trade value updates with better pricing"""
    updated_trades = []
    total_value = 0
    
    for trade in session['trades']:
        if trade['status'] == 'open':
            # Get current option price
            current_price = get_accurate_option_price(
                trade['symbol'], trade['expiration'], trade['strike'], trade['type']
            )
            
            # Get current underlying price
            market_data = get_enhanced_market_data(trade['symbol'])
            underlying_price = market_data['price'] if market_data else None
            
            trade['current_price'] = current_price
            trade['underlying_price'] = underlying_price
            
            if current_price is not None and underlying_price is not None:
                # Calculate P&L
                trade['unrealized_pl'] = round(
                    (current_price - trade['entry_price']) * trade['quantity'] * 100, 2
                )
                
                # Calculate intrinsic and extrinsic value
                if trade['type'].lower() == 'call':
                    intrinsic = max(0, underlying_price - trade['strike'])
                else:
                    intrinsic = max(0, trade['strike'] - underlying_price)
                
                trade['intrinsic_value'] = round(intrinsic, 2)
                trade['extrinsic_value'] = round(current_price - intrinsic, 2)
                
                # Calculate delta and other Greeks (simplified)
                S = underlying_price
                K = trade['strike']
                T = max((datetime.strptime(trade['expiration'], '%Y-%m-%d').date() - 
                        datetime.now().date()).days / 365.0, 0.001)
                
                market_data = get_enhanced_market_data(trade['symbol'])
                if market_data:
                    sigma = market_data['volatility']
                    r = get_risk_free_rate()
                    
                    try:
                        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                        
                        if trade['type'].lower() == 'call':
                            delta = norm.cdf(d1)
                        else:
                            delta = norm.cdf(d1) - 1
                        
                        trade['delta'] = round(delta, 3)
                    except:
                        trade['delta'] = None
                
                total_value += current_price * trade['quantity'] * 100
                
            else:
                trade['unrealized_pl'] = None
                trade['intrinsic_value'] = None
                trade['extrinsic_value'] = None
                trade['delta'] = None
        
        updated_trades.append(trade)
    
    session['trades'] = updated_trades
    session.modified = True
    return total_value

@app.before_request
def initialize_trades():
    """Initialize session variables"""
    if 'trades' not in session:
        session['trades'] = []
    if 'portfolio_value' not in session:
        session['portfolio_value'] = 10000.0
    if 'trade_history' not in session:
        session['trade_history'] = []
    if 'symbol' not in session:
        session['symbol'] = None
    if 'expirations' not in session:
        session['expirations'] = []
    if 'expiration' not in session:
        session['expiration'] = None
    if 'option_type' not in session:
        session['option_type'] = None
    if 'strikes' not in session:
        session['strikes'] = []
    
    session.modified = True

def ensure_str(val):
    if isinstance(val, list):
        return val[0]
    return val

@paper_trading.route('/trade', methods=['GET', 'POST'])
def trade():
    """Enhanced trading route with real market data"""
    error = None
    
    if request.method == 'POST':
        if 'reset' in request.form:
            session.pop('symbol', None)
            session.pop('expirations', None)
            session.pop('expiration', None)
            session.pop('option_type', None)
            session.pop('strikes', None)
            session.modified = True
            return redirect(url_for('paper_trading.trade'))
        
        if 'load_expirations' in request.form:
            symbol = request.form.get('symbol', '').strip().upper()
            if not symbol:
                error = "Please enter a valid symbol."
            else:
                # Verify symbol and get option chains
                market_data = get_enhanced_market_data(symbol)
                if not market_data:
                    error = f"Invalid symbol: {symbol}. No market data available."
                else:
                    option_chains = get_real_option_chains(symbol)
                    if not option_chains:
                        error = f"No options available for {symbol}. Try a different symbol."
                    else:
                        expirations = list(option_chains.keys())
                        session['symbol'] = symbol
                        session['expirations'] = expirations
                        session['expiration'] = None
                        session['option_type'] = None
                        session['strikes'] = []
                        session.modified = True
                        print(f"Loaded {len(expirations)} expirations for {symbol}")
        
        elif 'load_strikes' in request.form:
            symbol = session.get('symbol')
            expiration = request.form.get('expiration')
            option_type = request.form.get('option_type')
            
            if symbol and expiration:
                strikes = get_available_strikes(symbol, expiration)
                session['expiration'] = expiration
                session['option_type'] = option_type
                session['strikes'] = strikes
                session.modified = True
                print(f"Loaded {len(strikes)} strikes for {symbol} exp {expiration}")
        
        elif 'submit_trade' in request.form:
            symbol = request.form.get('symbol', '').strip().upper()
            option_type = request.form.get('type', '').lower()
            
            try:
                strike = float(request.form.get('strike'))
                quantity = int(request.form.get('quantity'))
                expiration = request.form.get('expiration')
            except (ValueError, TypeError):
                error = "Invalid strike price, quantity, or expiration date."
            
            if not error:
                if not all([symbol, option_type, strike, quantity, expiration]):
                    error = "All fields are required."
                elif quantity <= 0:
                    error = "Quantity must be positive."
                elif strike <= 0:
                    error = "Strike price must be positive."
                elif option_type not in ['call', 'put']:
                    error = "Invalid option type."
                else:
                    # Validate expiration date
                    try:
                        today = datetime.now().date()
                        exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                        if exp_date <= today:
                            error = "Expiration date must be in the future."
                    except ValueError:
                        error = "Invalid expiration date format."
                    
                    if not error:
                        # Get accurate option price
                        entry_price = get_accurate_option_price(symbol, expiration, strike, option_type)
                        if entry_price is None:
                            error = f"Unable to price {symbol} {option_type} {strike} expiring {expiration}."
                        else:
                            cost = entry_price * quantity * 100
                            if cost > session['portfolio_value']:
                                error = f"Insufficient funds. Need ${cost:.2f}, have ${session['portfolio_value']:.2f}."
                            else:
                                # Create trade
                                trade = {
                                    'id': str(uuid.uuid4()),
                                    'symbol': symbol,
                                    'type': option_type,
                                    'strike': strike,
                                    'quantity': quantity,
                                    'expiration': expiration,
                                    'entry_price': entry_price,
                                    'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'status': 'open',
                                    'current_price': entry_price,
                                    'unrealized_pl': 0.0,
                                    'underlying_price': None,
                                    'intrinsic_value': None,
                                    'extrinsic_value': None,
                                    'delta': None
                                }
                                
                                session['trades'].append(trade)
                                session['portfolio_value'] -= cost
                                
                                # Add to trade history
                                session['trade_history'].append({
                                    'type': 'open',
                                    'trade_id': trade['id'],
                                    'symbol': symbol,
                                    'option_type': option_type,
                                    'strike': strike,
                                    'quantity': quantity,
                                    'price': entry_price,
                                    'time': trade['entry_time'],
                                    'amount': -cost,
                                    'profit': None
                                })
                                
                                session.modified = True
                                print(f"Trade executed: {trade}")
    
    # Update all open positions
    open_positions_value = update_trade_values_enhanced()
    portfolio_total = session['portfolio_value'] + open_positions_value
    
    # Get current session data
    symbol = ensure_str(session.get('symbol', ''))
    expirations = session.get('expirations', [])
    strikes = session.get('strikes', [])
    option_type = session.get('option_type', '')
    expiration = session.get('expiration', '')
    
    # Get current market data for display
    market_info = None
    if symbol:
        market_info = get_enhanced_market_data(symbol)
    
    return render_template(
        'trade.html',
        trades=session.get('trades', []),
        error=error,
        expirations=expirations,
        symbol=symbol,
        strikes=strikes,
        option_type=option_type,
        expiration=expiration,
        portfolio_value=session.get('portfolio_value', 10000.0),
        portfolio_total=portfolio_total,
        trade_history=session.get('trade_history', []),
        market_info=market_info,
        is_market_open=is_market_open()
    )

@paper_trading.route('/close_trade/<trade_id>', methods=['POST'])
def close_trade(trade_id):
    """Enhanced trade closing with accurate pricing"""
    for trade in session['trades']:
        if trade['id'] == trade_id and trade['status'] == 'open':
            exit_price = get_accurate_option_price(
                trade['symbol'], trade['expiration'], trade['strike'], trade['type']
            )
            
            if exit_price is None:
                # If can't get price, use current price from trade data
                exit_price = trade.get('current_price', trade['entry_price'])
            
            proceeds = exit_price * trade['quantity'] * 100
            profit = round((exit_price - trade['entry_price']) * trade['quantity'] * 100, 2)
            
            # Update trade
            trade['exit_price'] = exit_price
            trade['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trade['profit'] = profit
            trade['status'] = 'closed'
            
            # Update portfolio
            session['portfolio_value'] += proceeds
            
            # Add to history
            session['trade_history'].append({
                'type': 'close',
                'trade_id': trade_id,
                'symbol': trade['symbol'],
                'option_type': trade['type'],
                'strike': trade['strike'],
                'quantity': trade['quantity'],
                'price': exit_price,
                'time': trade['exit_time'],
                'amount': proceeds,
                'profit': profit
            })
            
            session.modified = True
            print(f"Closed trade {trade_id}: Exit price=${exit_price}, Profit=${profit}")
            break
    
    return redirect(url_for('paper_trading.trade'))

@paper_trading.route('/refresh_prices', methods=['POST'])
def refresh_prices():
    """Force refresh of all market data"""
    # Clear caches to force refresh
    global MARKET_DATA_CACHE, OPTION_CHAIN_CACHE
    MARKET_DATA_CACHE.clear()
    OPTION_CHAIN_CACHE.clear()
    
    # Update all positions
    update_trade_values_enhanced()
    
    return redirect(url_for('paper_trading.trade'))

app.register_blueprint(paper_trading)

if __name__ == '__main__':
    app.run(debug=True)