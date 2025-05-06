import nextcord
from generate_text import generate_text
from nextcord.ext import tasks

GUILD_ID = 1162056243125432391

bot = nextcord.Client()
channel_id = 1369273668953440267 #slop general

@bot.event
async def on_ready():
    print("hi")
    rand_say.start()

@bot.event
async def on_message(message: nextcord.Message):
    if bot.user in message.mentions:
        await message.reply(generate_text(message.content))

@tasks.loop(minutes=30)
async def rand_say():
    channel = bot.get_channel(channel_id)
    msg = (await channel.history(limit=1).flatten())[0]

    if msg == None: return 
    if msg.author == bot.user: return
    if msg.content == "": return

    await channel.send(generate_text(msg.content))



bot.run('nuh uh')