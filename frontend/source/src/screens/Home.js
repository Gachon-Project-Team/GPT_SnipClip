import React from 'react'
import { Container } from '../components/Styles/Container/Container.style'
import { Header, MainHeader } from '../components/Styles/Header/Header.style'
import KeywordInput from '../components/KeywordInput'
import NavBarElements from '../components/NavBar/NavBarElements'


export const Home = () => {
  return (
    <Container>
      <Header>
        <MainHeader>
          <KeywordInput />
        </MainHeader>
      </Header>
    </Container>
  )
}
