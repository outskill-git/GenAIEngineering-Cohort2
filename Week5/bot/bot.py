"""Discord Bot using nextcord"""
import asyncio
import json
import time
import logging
import os
import platform
# import random
import sys

import aiosqlite
# import discord
import nextcord
from nextcord.ext import commands

from helpers.utils import load_config, load_cogs

config = load_config()
intents = nextcord.Intents.all()
intents.messages = True  # To receive message events
intents.message_content = True  # To read message content


class LoggingFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors and styles to log messages."""
    # Colors
    black = "\x1b[30m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    gray = "\x1b[38m"
    # Styles
    reset = "\x1b[0m"
    bold = "\x1b[1m"

    COLORS = {
        logging.DEBUG: gray + bold,
        logging.INFO: blue + bold,
        logging.WARNING: yellow + bold,
        logging.ERROR: red,
        logging.CRITICAL: red + bold
    }

    def format(self, record):
        log_color = self.COLORS[record.levelno]
        format = "(black){asctime}(reset) (levelcolor){levelname:<8}(reset) (green){name}(reset) {message}"
        format = format.replace("(black)", self.black + self.bold)
        format = format.replace("(reset)", self.reset)
        format = format.replace("(levelcolor)", log_color)
        format = format.replace("(green)", self.green + self.bold)
        formatter = logging.Formatter(format, "%Y-%m-%d %H:%M:%S", style="{")
        return formatter.format(record)


logger = logging.getLogger("discord_bot")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(LoggingFormatter())
# File handler
file_handler = logging.FileHandler(
    filename="discord.log", encoding="utf-8", mode="w")
file_handler_formatter = logging.Formatter(
    "[{asctime}] [{levelname:<8}] {name}: {message}", "%Y-%m-%d %H:%M:%S", style="{")
file_handler.setFormatter(file_handler_formatter)

# Add the handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


bot = commands.Bot(intents=intents, )

bot.logger = logger
bot.config = config


async def init_db():
    async with aiosqlite.connect(f"{os.path.realpath(os.path.dirname(__file__))}/database/database.db") as db:
        with open(f"{os.path.realpath(os.path.dirname(__file__))}/database/schema.sql") as file:
            await db.executescript(file.read())
        await db.commit()


@bot.event
async def on_ready() -> None:
    """
    The code in this event is executed when the bot is ready.
    """
    bot.logger.info("Loading Cogs")
    await load_cogs(bot, config)
    bot.logger.info("Logged in as %s", bot.user.name)
    bot.logger.info("nextcord API version: %s", nextcord.__version__)
    bot.logger.info(f"Python version: {platform.python_version()}")
    bot.logger.info(
        f"Running on: {platform.system()} {platform.release()} ({os.name})")
    bot.logger.info("-------------------")
    if config["sync_commands_globally"]:
        bot.logger.info("Syncing commands globally...")
        await bot.sync_all_application_commands()
        bot.logger.info("Done")
    bot.logger.info(f"Bot Ready: {bot.is_ready()}")


@bot.event
async def on_message(message: nextcord.Message) -> None:
    """
    This event is triggered when a message is sent in a channel the bot
    has access to.
    """
    bot.logger.info(message)
    if message.author == bot.user or message.author.bot:
        return

    await bot.process_commands(message)


@bot.slash_command(description="Replies with pong!")
async def ping(interaction: nextcord.Interaction):
    await interaction.send("Pong!", ephemeral=True)


asyncio.run(init_db())
bot.run(config["token"],)
