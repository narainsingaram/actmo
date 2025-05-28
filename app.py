from flask import Flask, render_template, request, jsonify # Added jsonify
import yfinance as yf
import ta
import datetime
from google import genai # Assuming this is google.ai.generativelanguage or similar
import markdown
from papertrade import paper_trading  # ✅ Import it here

app = Flask(__name__)

app.secret_key = 'dev_secret_key_123'

# Replace with your actual API key
GEMINI_API_KEY = "AIzaSyCvjrlF-nctXVwZAwzBCYj2gryjT-VxYAI" # Keep your actual key secure

app.register_blueprint(paper_trading)

# New route to fetch current stock price
@app.route('/get_current_price/<symbol_ticker>')
def get_current_price(symbol_ticker):
    try:
        stock = yf.Ticker(symbol_ticker.upper())
        # Attempt to get recent historical data for the last closing price
        data = stock.history(period="2d") # Fetch last two days to get the most recent close

        if not data.empty:
            current_price = data['Close'].iloc[-1]
        else:
            # Fallback to stock.info if history is empty (e.g., for some specific symbols or circumstances)
            stock_info = stock.info
            current_price = stock_info.get('currentPrice') or \
                            stock_info.get('regularMarketPrice') or \
                            stock_info.get('open') or \
                            stock_info.get('previousClose') # Add more fallbacks if needed

        if current_price is None:
            return jsonify({"error": f"Could not determine current price for {symbol_ticker}"}), 404

        return jsonify({"symbol": symbol_ticker, "currentPrice": float(current_price)})
    except Exception as e:
        # yfinance can sometimes fail to fetch data for various reasons (invalid symbol, network issues, etc.)
        print(f"Error fetching price for {symbol_ticker}: {e}") # Log error for debugging
        return jsonify({"error": f"Error fetching price data for {symbol_ticker}. Please ensure the symbol is correct and try again."}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    analysis = None
    error = None
    ai_recommendation_html = None
    # Retain form values for symbol, etc. for when the page reloads after chart display
    form_data = request.form if request.method == 'POST' else {}


    if request.method == 'POST':
        try:
            symbol = request.form['symbol']
            strike = float(request.form['strike'])
            premium = float(request.form['premium'])
            dte = int(request.form['dte'])
            option_type = request.form['type'].lower()

            end = datetime.datetime.today()
            fetch_days = 150
            start = end - datetime.timedelta(days=fetch_days)
            stock_ticker_obj = yf.Ticker(symbol) # Renamed to avoid conflict
            hist = stock_ticker_obj.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

            if hist.empty or len(hist) < 3:
                error = "Not enough data for analysis. Try a symbol with more history."
                return render_template('index.html', error=error, request_form=request.form)

            hist = hist.tail(150)

            ema_short, ema_mid, ema_long = 21, 50, 100
            rsi_win = 14
            atr_win = 14
            macd_fast = 12
            macd_slow = 26
            macd_sig = 9
            sma_win = 21

            n = len(hist)
            ema_short = min(ema_short, n)
            ema_mid   = min(ema_mid, n)
            ema_long  = min(ema_long, n)
            rsi_win   = min(rsi_win, n)
            atr_win   = min(atr_win, n)
            macd_fast = min(macd_fast, n)
            macd_slow = min(macd_slow, n)
            macd_sig  = min(macd_sig, n)
            sma_win   = min(sma_win, n)

            hist[f'EMA{ema_short}'] = ta.trend.EMAIndicator(hist['Close'], window=ema_short).ema_indicator()
            hist[f'EMA{ema_mid}']   = ta.trend.EMAIndicator(hist['Close'], window=ema_mid).ema_indicator()
            hist[f'EMA{ema_long}']  = ta.trend.EMAIndicator(hist['Close'], window=ema_long).ema_indicator()
            hist[f'SMA{sma_win}'] = ta.trend.SMAIndicator(hist['Close'], window=sma_win).sma_indicator()
            hist['RSI'] = ta.momentum.RSIIndicator(hist['Close'], window=rsi_win).rsi()
            hist['ATR'] = ta.volatility.AverageTrueRange(hist['High'], hist['Low'], hist['Close'], window=atr_win).average_true_range()
            macd = ta.trend.MACD(hist['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_sig)
            hist['MACD'] = macd.macd()
            hist['MACD_signal'] = macd.macd_signal()

            required_cols = [
                f'EMA{ema_short}', f'EMA{ema_mid}', f'EMA{ema_long}', f'SMA{sma_win}',
                'RSI', 'ATR', 'MACD', 'MACD_signal'
            ]
            valid_hist = hist.dropna(subset=required_cols)
            if valid_hist.empty:
                error = "Not enough data to compute indicators. Try a different symbol or wait for more data."
                return render_template('index.html', error=error, request_form=request.form)
            
            last_row = valid_hist.iloc[-1]
            current_price = last_row['Close']
            rsi = last_row['RSI']
            ema1 = last_row[f'EMA{ema_short}']
            ema2 = last_row[f'EMA{ema_mid}']
            ema3 = last_row[f'EMA{ema_long}']
            sma_val = last_row[f'SMA{sma_win}']
            atr = last_row['ATR']
            macd_val = last_row['MACD']
            macd_signal_val = last_row['MACD_signal']

            break_even = strike + premium if option_type == 'call' else strike - premium
            distance_to_breakeven = abs(current_price - break_even)
            trend_score = 1 if ema1 > ema2 > ema3 else -1
            momentum_score = 1 if rsi > 55 else (-1 if rsi < 45 else 0)
            macd_score = 1 if macd_val > macd_signal_val else -1
            volatility_score = -1 if atr / current_price > 0.05 else 1
            total_score = trend_score + momentum_score + macd_score + volatility_score

            summary = f"""
You are a bold and tactical stock trading assistant. Your job is to give decisive and actionable advice based on technical indicators.
You can highlight risks, but avoid generic phrases like 'avoid for now' unless there's truly no opportunity. VERY BOLD, DECISIVE, and ACTIONABLE advice.
Suggest possible entry and exit zones, trend direction, and confidence level. Be concise, direct, and intelligent.
You are a skilled and unbiased options trading assistant. Based on the data below, provide a brief and **balanced** analysis. Focus on trend, momentum, volatility, and breakeven distance. Then give a **clear, actionable recommendation**: should the user take this trade or not?

If the setup is not ideal, offer a constructive alternative (e.g., different strike, expiry, or option type). 

Give a genuine answer based on analyzing everything. Don't be overly conservative. Be bold and decisive. Give an exact answer on why.

Please analyze the indicators and give a clear actionable insight.
State whether you recommend buying, selling, or waiting — and why.
Include a confidence level (High / Medium / Low) and potential upside/downside in percentage terms and have a confidence percentage on this like what % confident you are

Stock symbol: {symbol}  
Option type: {option_type}  
Strike price: {strike}  
Premium: {premium}  
Days to expiration: {dte}  
Current stock price: {current_price:.2f}  
Breakeven price: {break_even:.2f} (distance from current price: {distance_to_breakeven:.2f})  

Technical indicators (based on latest close):  
- EMA{ema_short}: {ema1:.2f}  
- EMA{ema_mid}: {ema2:.2f}  
- EMA{ema_long}: {ema3:.2f}  
- SMA{sma_win}: {sma_val:.2f}  
- RSI ({rsi_win}): {rsi:.2f}  
- ATR ({atr_win}): {atr:.2f}  
- MACD (fast={macd_fast}, slow={macd_slow}, signal={macd_sig}): {macd_val:.2f}  
- MACD signal: {macd_signal_val:.2f}  

Overall sentiment score based on indicators: {total_score} (scale: -4 to +4)

Return a markdown response with (BULLET POINT SECTIONS):
- **Trend Analysis**
- **Momentum Assessment**
- **Volatility Note**
- **Breakeven Analysis**
- **Final Recommendation**: *Take trade* / *Avoid for now* / *Consider alternate setup*

Give an elaborated explanation that is extremely detailed and actionable. 

Based on all the recent news and potential future news or things that could happen in the future about {symbol} (symbol) stock, give a good direction the user can take in terms of the trade
and elucidate why. Access real-time news and previous sources and provide information from where they are from.

MARKDOWN ALL IMPORANT INFORMATION. USE ITALICS, CODE BLOCKS, BOLD TEXT ETC. Space out make it readable. MOST IMPORTANTLY INCLUDED STYLED BULLETPOINTS!!!

Should the user consider entering this trade now, wait for a better setup, or adjust parameters like strike or DTE?
"""
            try:
                # Ensure your GEMINI_API_KEY is correctly configured for this client usage
                client = genai.Client(api_key=GEMINI_API_KEY) 
                response = client.models.generate_content( # Verify this client and method if issues arise
                    model="gemini-2.0-flash", # Verify model name if issues arise
                    contents=summary
                )
                ai_recommendation = response.text
                ai_recommendation_html = markdown.markdown(ai_recommendation)
            except Exception as ai_err:
                print(f"AI analysis error: {ai_err}") # Log AI error
                ai_recommendation_html = f"<p class='text-red-400'><b>AI analysis failed:</b> {ai_err}. Please check API key and model configuration.</p>"


            analysis = [
                f"Stock symbol: {symbol}",
                f"Option type: {option_type}",
                f"Strike price: {strike}",
                f"Premium: {premium}",
                f"Days to expiration: {dte}",
                f"Current price (at analysis): {current_price:.2f}", # Clarify this is price at analysis time
                f"Breakeven price: {break_even:.2f} (distance: {distance_to_breakeven:.2f})",
                f"EMA{ema_short}: {ema1:.2f}",
                f"EMA{ema_mid}: {ema2:.2f}",
                f"EMA{ema_long}: {ema3:.2f}",
                f"SMA{sma_win}: {sma_val:.2f}",
                f"RSI ({rsi_win}): {rsi:.2f}",
                f"ATR ({atr_win}): {atr:.2f}",
                f"MACD: {macd_val:.2f}",
                f"MACD signal: {macd_signal_val:.2f}",
            ]

            if dte < 7:
                analysis.append("⚠️ Warning: Very short DTE. Technical indicators may be less reliable.")
            
            # Pass request.form to keep form populated
            return render_template('index.html',
                                   analysis=analysis,
                                   ai_recommendation_html=ai_recommendation_html,
                                   request_form=request.form, # Pass the whole form
                                   current_price_at_analysis=round(current_price, 2)) # Distinguish this price

        except Exception as e:
            error = f"Error processing request: {str(e)}"
            print(f"General error: {e}") # Log general error
            # Pass request.form to keep form populated even on error
            return render_template('index.html', error=error, request_form=request.form)

    # For GET requests, or if POST fails before form processing
    return render_template('index.html', error=error, request_form=form_data, ai_recommendation_html=ai_recommendation_html, analysis=analysis)

@app.route('/info')
def info():
    return render_template('info.html')  # Renders about page

if __name__ == '__main__':
    app.run(debug=True)