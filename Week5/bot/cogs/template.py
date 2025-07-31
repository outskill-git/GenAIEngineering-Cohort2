import nextcord
from nextcord.ext import commands
from helpers.utils import load_config
config = load_config()


class Template(commands.Cog):
    def __init__(
        self,
        bot: commands.Bot
    ) -> None:
        super().__init__()
        self.bot = bot

    @nextcord.slash_command(
        guild_ids=[config['guild_id']],
        description="Sample")
    async def sample(
            self,
            interaction: nextcord.Interaction
    ):
        await interaction.response.defer()
        await interaction.followup.send(
            content="This is a slash command in a cog!"
            )


def setup(bot):
    bot.add_cog(Template(bot))
