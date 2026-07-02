import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# from langchain_openai import ChatOpenAI\
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """The Taj Mahal (/ˌtɑːdʒ məˈhɑːl, ˌtɑːʒ -/ TAHJ mə-HAHL, TAHZH -⁠; Hindustani: [t̪ɑːd͡ʒ ˈmɛɦ(ɛ)l]; lit. 'Crown of the Palace') is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India. It was commissioned in 1631 by the fifth Mughal emperor, Shah Jahan (r. 1628–1658), to house the tomb of his late wife, Mumtaz Mahal; it also houses the tomb of Shah Jahan himself. The tomb is the centrepiece of a 17-hectare (42-acre) complex, which includes a mosque and a guest house, and is set in formal gardens bounded on three sides by a crenellated wall.

Construction of the mausoleum was completed in 1648, while work on other parts of the complex continued for another five years. The first ceremony held at the mausoleum was an observance by Shah Jahan, on 6 February 1643, of the 12th anniversary of the death of Mumtaz Mahal. The Taj Mahal complex is believed to have been completed in its entirety in 1653 at a cost estimated at the time to be around ₹32 million, which in 2015 would be approximately ₹52.8 billion (US$827 million).[4]

The building complex incorporates the design traditions of Indo-Islamic and Mughal architecture. It employs symmetrical constructions with the usage of various shapes and symbols. While the mausoleum is constructed of white marble inlaid with semi-precious stones, red sandstone was used for other buildings in the complex similar to contemporary Mughal-era buildings. The construction project employed more than 20,000 workers and artisans under the guidance of a board of architects led by Ustad Ahmad Lahori, the emperor's court architect. The complex was designed and executed by a multinational board of artisans and supervisors, including Ottoman dome designer Ismail Afandi; Persian architects Ustad Isa, Isa Muhammad Effendi, and Puru; chief calligrapher Amanat Khan Shirazi; finial caster Qazim Khan; and masonry supervisors Muhammad Hanif, Mir Abdul Karim, and Mukkarimat.[5]

The Taj Mahal was designated as a UNESCO World Heritage Site in 1983 for being "the jewel of Islamic art in India and one of the universally admired masterpieces of the world's heritage". It is regarded as one of the best examples of Mughal architecture and a symbol of Indian history. The Taj Mahal is a major tourist attraction and attracts more than five million visitors a year. In 2007, it was declared a winner of the New 7 Wonders of the World initiative. The Taj Mahal and its setting, surrounding grounds, and structures are a Monument of National Importance, administered by the Archaeological Survey of India.[6]"""
    system_template = """give me the short summery of information :{information}
    and outline two interesting fact about it"""

    system_prompt = PromptTemplate(
        template=system_template, input_variables=["information"]
    )
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
    chain = system_prompt | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
