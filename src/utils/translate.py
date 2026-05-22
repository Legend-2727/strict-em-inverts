import asyncio
from googletrans import Translator


async def translate_string_async(s, src_lang, dest_lang):
    """
    Translates a string asynchronously.
    """
    async with Translator() as translator:
        # The 'await' keyword is required for the newer async versions of googletrans
        result = await translator.translate(s, src=src_lang, dest=dest_lang)
        return result.text


async def translate_list_throttled_async(samples, src_lang, dest_lang, delay=1):
    """
    Translates a list of strings with a 1-second delay between requests
    to avoid rate limiting, handling the async nature of the library.
    """
    async with Translator() as translator:
        translated_results = []

        for sample in samples:
            # Await the translation
            result = await translator.translate(sample, src=src_lang, dest=dest_lang)
            translated_results.append(result.text)

            # Use asyncio.sleep instead of time.sleep to avoid blocking the event loop
            await asyncio.sleep(delay)

        return translated_results

# # --- How to run this code ---
# if __name__ == "__main__":
#     # You must use asyncio.run() to execute the async function

#     # Example for single string
#     result = asyncio.run(translate_string_async('Hello', 'english', 'hindi'))
#     print(f"Single result: {result}")

#     # Example for list
#     sample_list = ["Hello", "World", "Good morning"]
#     results = asyncio.run(translate_list_throttled_async(sample_list, 'english', 'hindi'))
#     print(f"Batch results: {results}")
