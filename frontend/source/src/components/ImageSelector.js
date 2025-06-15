import { useState } from 'react';
import { Modal, Form } from 'react-bootstrap';
import {
  SelectorWrapper,
  ImageBlock,
  ClickableImage,
  RadioWrapper,
  RadioLabel,
  ModalImage,
} from './Styles/Container/ImageSelector.styles';

function ImageSelector({ realImageUrl, aiImageUrl, selected, onChange, groupName }) {
  const [show, setShow] = useState(false);
  const [modalImage, setModalImage] = useState('');

  const handleImageClick = (src) => {
    setModalImage(src);
    setShow(true);
  };

  return (
    <>
      <SelectorWrapper>
        <ImageBlock>
          <ClickableImage
            src={realImageUrl}
            alt="실제 이미지"
            onClick={() => handleImageClick(realImageUrl)}
          />
          <RadioWrapper>
            <Form.Check.Input
              type="radio"
              name={groupName}
              id="real"
              value="real"
              checked={selected === 'real'}
              onChange={onChange}
            />
            <RadioLabel htmlFor="real">실제 이미지</RadioLabel>
          </RadioWrapper>
        </ImageBlock>

        <ImageBlock>
          <ClickableImage
            src={aiImageUrl}
            alt="AI 이미지"
            onClick={() => handleImageClick(aiImageUrl)}
          />
          <RadioWrapper>
            <Form.Check.Input
              type="radio"
              name={groupName}
              id="ai"
              value="ai"
              checked={selected === 'ai'}
              onChange={onChange}
            />
            <RadioLabel htmlFor="ai">AI</RadioLabel>
          </RadioWrapper>
        </ImageBlock>
      </SelectorWrapper>

      <Modal show={show} onHide={() => setShow(false)} centered size="lg">
        <Modal.Body style={{ textAlign: 'center' }}>
          <ModalImage src={modalImage} alt="확대 이미지" />
        </Modal.Body>
      </Modal>
    </>
  );
}

export default ImageSelector;